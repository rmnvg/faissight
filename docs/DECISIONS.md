# Decisions log

One line per non-obvious technical decision.

- Dev environment pinned to Python 3.12 (`.python-version`); `requires-python>=3.10`. System Python 3.14 may lack faiss-cpu wheels.
- `httpx` added to the `dev` extra: FastAPI's `TestClient` requires it.
- `__version__` is read from installed package metadata so `pyproject.toml` is the single source of truth.
- Frontend pinned to React 18 (per plan) although the Vite template defaults to 19.
- Vite `base: './'` (relative asset URLs) so the built UI works under path-prefixing proxies (Jupyter/Colab/HF Spaces).
- Tailwind v4 via `@tailwindcss/vite` plugin (no PostCSS/tailwind.config needed).
- `static/` is gitignored but force-included in wheel and sdist via hatch `artifacts`; the frontend must be built before `uv build`.
- Synthetic builders live in `examples/make_synthetic.py`; `tests/conftest.py` imports it by path so demos and tests share one code path (at smaller n/nlist).
- mypy has no `python_version` pin: numpy>=2.5 stubs use 3.12-only syntax. Each CI matrix job typechecks against its own interpreter instead.
- `LoadedIndex.index` holds the owning root object; `core`/`ivf` are non-owning `downcast_index` views that dangle if the root is garbage-collected.
- IVF core is found by unwrapping PreTransform/IDMap/Refine ourselves and checking `isinstance(core, IndexIVF)`, not `extract_index_ivf`, so `kind`, `core` and `ivf` always agree.
- `d` is the input dimension (what queries use); `core_d` is the dimension after any PreTransform (what centroids/codes use).
- Binary, GPU, FastScan, non-L2/IP metrics and other exotic indexes load as `UNSUPPORTED` with a reason instead of raising.
- IVF centroids of a PreTransform index are returned in the transformed space (`core_d`); anything comparing them to raw vectors must apply the transform first.
- `list_members`/`assignments` read ids via `invlists.get_ids` directly instead of `inspect_tools.get_invlist`, which also copies every list's codes.
- `imbalance_factor` returns 0.0 for an empty index (FAISS returns NaN, which isn't valid JSON).
- Search parameters go through `SearchParametersIVF/HNSW` per call, never by mutating `nprobe`/`efSearch`. `IndexRefine` needs them wrapped in `IndexRefineSearchParameters` with the index's own `k_factor` (its default of 1 would change results).
- Self-exclusion searches k+1 and drops the query's own id, so the caller still gets k results.
- New `core/vectors.py` (not in the original layout): `VectorSource` holds input-space vectors sorted by user id, whether raw or reconstructed; ground truth, query-by-id, projection and PQ error all share it.
- Reconstruction enables a direct map on IVF indexes (array for sequential ids, hashtable for `add_with_ids` ids). This mutates the index object but not its search results (tested).
- Reconstructed vectors for PreTransform indexes are mapped back to input space with `reverse_transform`, which is lossy for PCA.
