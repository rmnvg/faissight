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
- `faissight.core` does not re-export the `search` function: it would shadow the `core.search` submodule (`from faissight.core import search` would return the function). Use `core.search.search(...)`.
- Projections are computed in core space (after PreTransform) so points, IVF centroids and queries share one space.
- PCA uses `eigh` on the d×d covariance rather than an SVD of the data (~5x faster for n >> d); component signs are normalised for determinism.
- UMAP queries are placed by inverse-distance-weighted kNN interpolation over the projected sample, not `umap.transform` (4.7 s for 10 queries on 20k points). Centroids use `.transform` once at fit time and are cached.
- Projection cache key = index sha1 (dir) + method + dims + max_points + seed + sha1 of the sampled vectors, so raw vs reconstructed vectors never collide. Stored as `.npz` (no pickle), written atomically. `FAISSIGHT_CACHE_DIR` / `XDG_CACHE_HOME` override the location.
- UMAP tests are marked `slow` (cold numba import ~35 s) and excluded by default; CI runs them with `-m "slow or not slow"`.
- `Session` lives at package level (not in `core`) but has no web imports; the server and `launch()` both wrap it.
- Session caches are computed on first use under an `RLock`. Building an IVF direct map during reconstruction is safe alongside searches: IVF search never reads the direct map, and building it only reads the inverted lists.
- A `--embedder` model name loads in a background job at startup (importing sentence-transformers takes ~15 s); the first text query waits for it. Query embeddings are normalised iff the raw vectors are unit-norm (or, without raw vectors, iff the metric is IP), unless overridden.
- The session resolves its cache directory at construction, not in the background thread, so env changes can't redirect a running job (tests caught stray writes to ~/.cache).
- Projection responses are columnar (`ids`, `x`, `y`, `z`, `list_nos`) with coordinates rounded to 4 decimals: smaller JSON than `[[x, y], ...]` and maps directly onto deck.gl attributes.
- `/api/search` takes an optional `projection: {method, dims}`; if that projection is ready, the response includes the query's coordinates. The UI can't place text queries itself, since it never sees vectors.
- Unknown `/api/*` paths return JSON 404s; all other unknown paths fall back to `index.html` for client-side routing. Without a built frontend, `/` serves a short "run npm run build" page.
- OpenAPI docs are served at `/api/docs` so they don't collide with SPA routes.
