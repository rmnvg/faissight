# faissight: Build Plan

> **faissight** (FAISS + insight): see inside your FAISS index, debug retrieval, tune recall.
>
> - PyPI package: `faissight`
> - Import name: `faissight`
> - CLI command: `faissight`
> - Repo: `github.com/rmnvg/faissight`
> - License: MIT

---

## 0. How to use this plan (instructions for Claude Code)

- Work **phase by phase, in order**. Do not start a phase until the previous phase's acceptance criteria pass.
- At the start, create a `CLAUDE.md` in the repo root summarising: project goal, repo layout, commands (`uv run pytest`, `uv run ruff check`, `npm run build`), and coding conventions from section 5.
- Make **one git commit per completed step** with a conventional commit message (`feat:`, `fix:`, `test:`, `docs:`, `chore:`).
- Write tests alongside code, not after. Every public function in `faissight.core` needs at least one test.
- Before adding any dependency not listed in section 4, stop and ask.
- When a FAISS API behaves differently than this plan assumes, trust the installed FAISS version, adapt, and leave a short comment explaining why.
- Keep a `docs/DECISIONS.md` log: one line per non-obvious technical decision.

---

## 1. Problem & positioning

### The problem
FAISS is the default vector index in countless RAG systems, but it has **no UI**. When retrieval is bad, engineers can't easily answer:
- Which cluster/cell did my query probe, and were the true nearest neighbours even in those cells?
- Is my IVF index badly imbalanced (a few giant lists, many empty ones)?
- How much accuracy did PQ compression cost me?
- What `nprobe` / `efSearch` gives me 95% recall at acceptable latency?
- Why did this irrelevant chunk rank #1 and the relevant one not appear?

### Prior art (mention honestly in README)
- **Feder** (zilliztech/feder): JS visualizer for Faiss IVF_Flat and HNSWlib indexes. Educational, shows clusters and search process. Limitations: Faiss support only for IVF_Flat, parses index files in JavaScript (memory-heavy), no recall/tuning or RAG debugging.
- **Embedding projectors** (TensorFlow Projector, Nomic Atlas, Renumics Spotlight, Arize Phoenix): visualise embeddings, not index internals.

### faissight's angle
A **Python-first retrieval debugger for FAISS**:
1. Works on real indexes (IVF-Flat, IVF-PQ, IVF-SQ, HNSW, Flat, with IDMap/PreTransform wrappers).
2. Explains **misses** against exact ground truth, not just shows pictures.
3. Tunes the **recall vs latency** trade-off interactively.
4. Attaches **chunk text** so it doubles as a RAG debugging tool.

### Goals (v0.1.0)
- `pip install "faissight[faiss-cpu]"` then `faissight serve my.index` opens a local web UI.
- `faissight.launch(index, ...)` works from Python and renders inline in Jupyter.
- IVF overview + query explorer + recall tuner + RAG text panel.
- HNSW graph view with animated search trace.
- PQ reconstruction-error view.
- Live hosted demo on Hugging Face Spaces.

### Non-goals (v0.1.0)
- GPU indexes (users can `faiss.index_gpu_to_cpu` first; document this).
- Binary indexes, LSH, IndexIVFFastScan internals (show "unsupported: basic stats only").
- Editing/rebuilding indexes from the UI.
- Multi-user hosting or auth.

---

## 2. Architecture

```
┌────────────────────────────── faissight (Python package) ──────────────────────────────┐
│                                                                                         │
│  CLI (Typer)            Python API                    Jupyter helper                    │
│  faissight serve/info   faissight.launch(...)         IFrame to local server            │
│        │                       │                             │                          │
│        └───────────────┬───────┴─────────────────────────────┘                          │
│                        ▼                                                                │
│                 Session (in-memory state: index, vectors, metadata, caches)             │
│                        │                                                                │
│        ┌───────────────┼───────────────────────────────┐                                │
│        ▼               ▼                               ▼                                │
│  core.loader     core.ivf / core.hnsw / core.pq   core.search (trace, ground truth,     │
│  (read, unwrap,  (extract structure, stats)       sweep)                                │
│   detect type)                                                                          │
│        │                                                                                │
│        ▼                                                                                │
│  core.projection (PCA default, UMAP optional, sampling, disk cache)                     │
│                        │                                                                │
│                        ▼                                                                │
│        FastAPI server (JSON API + serves built frontend from package static/)           │
└────────────────────────┬────────────────────────────────────────────────────────────────┘
                         │ HTTP (localhost)
                         ▼
          React + TypeScript frontend (Vite build, bundled into the wheel)
          deck.gl for large scatter/graph views, Recharts for charts
```

Key principle: **all FAISS work happens in Python.** The frontend only receives compact JSON (projected 2D/3D coordinates, ids, small stats). Never ship raw high-dimensional vectors to the browser.

---

## 3. Repository layout

```
faissight/
├── CLAUDE.md
├── README.md
├── LICENSE
├── pyproject.toml
├── .pre-commit-config.yaml
├── .github/workflows/
│   ├── ci.yml                # lint, typecheck, test (py 3.10–3.12), frontend build
│   └── release.yml           # build frontend → build wheel → publish to PyPI (trusted publishing)
├── src/faissight/
│   ├── __init__.py           # exports launch, Session, __version__
│   ├── cli.py                # Typer app: serve, info, sweep
│   ├── session.py            # Session class holding loaded state
│   ├── jupyter.py            # notebook display helper
│   ├── core/
│   │   ├── loader.py         # read_index, unwrap wrappers, detect IndexKind
│   │   ├── types.py          # dataclasses / pydantic models shared across core
│   │   ├── ivf.py            # centroids, list sizes, imbalance, members
│   │   ├── hnsw.py           # graph extraction, levels, neighbours
│   │   ├── pq.py             # PQ/SQ reconstruction error
│   │   ├── search.py         # search w/ params, ground truth, IVF trace, miss reasons
│   │   ├── hnsw_trace.py     # Python re-implementation of HNSW greedy search for tracing
│   │   ├── sweep.py          # recall@k and latency vs nprobe / efSearch
│   │   ├── projection.py     # PCA/UMAP, sampling, caching
│   │   ├── metadata.py       # load jsonl/csv/parquet chunk metadata
│   │   └── embed.py          # optional text → vector (sentence-transformers or callable)
│   ├── server/
│   │   ├── app.py            # FastAPI app factory
│   │   ├── routes.py         # API endpoints (section 7)
│   │   └── schemas.py        # pydantic response models
│   └── static/               # built frontend (gitignored, produced by build)
├── frontend/
│   ├── package.json
│   ├── vite.config.ts        # build output → ../src/faissight/static
│   └── src/
│       ├── api/              # typed fetch client
│       ├── views/            # Overview, ClusterMap, QueryExplorer, Tuner, HnswGraph, Quantization
│       ├── components/
│       └── App.tsx
├── examples/
│   ├── make_synthetic.py     # gaussian-blob indexes of each type (for tests + quick demos)
│   ├── make_rag_demo.py      # real text dataset → embeddings → several indexes + chunks.jsonl
│   └── notebook_demo.ipynb
├── tests/
│   ├── conftest.py           # fixtures build small indexes in tmp dirs
│   ├── test_loader.py
│   ├── test_ivf.py
│   ├── test_hnsw.py
│   ├── test_search.py
│   ├── test_hnsw_trace.py
│   ├── test_sweep.py
│   ├── test_projection.py
│   └── test_api.py
├── deploy/hf-space/
│   ├── Dockerfile
│   └── README.md             # HF Space card metadata
└── docs/
    ├── DECISIONS.md
    └── screenshots/
```

---

## 4. Tech stack & dependencies

**Python (3.10+)**
- Tooling: `uv` (env + build), `hatchling` (build backend), `ruff` (lint + format), `mypy` (strict on `faissight.core`), `pytest`, `pytest-cov`, `pre-commit`.
- Core deps: `numpy`, `fastapi`, `uvicorn[standard]`, `pydantic>=2`, `typer`, `rich`.
- Extras:
  - `faissight[faiss-cpu]` → `faiss-cpu>=1.8`. FAISS is **not** a hard dependency, because conda and GPU users install FAISS differently and a pip `faiss-cpu` would clash. On import failure, raise a clear error telling the user how to install it.
  - `faissight[umap]` → `umap-learn` (heavy; PCA is the default projection).
  - `faissight[text]` → `sentence-transformers` (text queries).
  - `faissight[parquet]` → `pyarrow`.
  - `faissight[all]` → everything above.
  - `faissight[dev]` → test/lint tools.

**Frontend**
- React 18 + TypeScript + Vite.
- Tailwind CSS.
- `deck.gl` (ScatterplotLayer, LineLayer, OrbitView) for cluster maps and HNSW graphs; handles 100k+ points via WebGL.
- `recharts` for bar/line charts (list sizes, recall curves).
- `@tanstack/react-query` for API state.
- No component library needed; keep UI clean and minimal.

---

## 5. Conventions

- `src/` layout; type hints everywhere; docstrings on public functions.
- `faissight.core` must **not** import FastAPI or anything web-related. It's a plain library usable without the server.
- All IDs exposed to the UI are the **user-facing ids** (after IDMap), never internal offsets. Keep a mapping in the Session.
- Distances: always report the metric (`L2` or `IP`) alongside values. For IP, "closer" = larger.
- Large arrays: never return more than `max_points` (default 50,000) points to the frontend; sample deterministically (seeded) and stratified by IVF list when applicable.
- Every expensive computation (projection, ground truth, sweep) is cached on the Session, and projections are also cached on disk at `~/.cache/faissight/<index_sha1>/`.
- Errors surface to the UI as structured JSON `{error_code, message, hint}`.

---

## 6. Inputs the tool accepts

| Input | Required | Formats | Purpose |
|---|---|---|---|
| FAISS index | yes | file path or in-memory `faiss.Index` | the thing being inspected |
| Raw vectors | optional but strongly recommended | `.npy` (float32, shape `n×d`, row i ↔ id i or aligned with `--ids`) | exact ground truth, PQ error |
| Ids for raw vectors | optional | `.npy` int64 | when row order ≠ ids |
| Metadata | optional | `.jsonl` / `.csv` / `.parquet` with an `id` column plus e.g. `text`, `source`, `title` | RAG debugging text |
| Embedder | optional | sentence-transformers model name (CLI) or Python callable `str -> np.ndarray` (API) | type text queries |
| Query vectors | optional | `.npy` | batch evaluation / sweep queries |

If raw vectors are **not** given: ground truth falls back to exact search over **reconstructed** vectors (`reconstruct_n` after `make_direct_map`). The UI must show a visible banner: "Ground truth computed on reconstructed vectors; PQ/SQ error is not measured."

If no query set is given for the sweep: sample 200 stored vectors as queries (seeded), and exclude each query's own id from its results.

---

## 7. API endpoints (FastAPI)

All under `/api`. Response models in `server/schemas.py`.

| Method | Path | Returns |
|---|---|---|
| GET | `/info` | index kind, class chain (wrappers), d, ntotal, metric, is_trained, params (nlist, nprobe, M, efSearch, PQ m/nbits…), which optional inputs are loaded |
| GET | `/ivf/lists` | per-list sizes, empty count, imbalance factor, min/median/max, top-20 largest lists |
| GET | `/ivf/list/{list_no}` | member ids (paginated), with metadata snippets |
| GET | `/projection?kind=points\|centroids&method=pca\|umap&dims=2\|3` | coordinates + ids + list assignment (sampled) |
| POST | `/search` | body: `{query: {id \| vector \| text}, k, nprobe?, efSearch?}` → results with distance, list_no, rank, metadata; plus ground truth, recall@k, and miss explanations |
| POST | `/trace/ivf` | probe order of cells with centroid distances, which were probed, where each true neighbour lives and the probe rank its cell would have needed |
| POST | `/trace/hnsw` | per-layer visited nodes in order, edges traversed, entry point, final candidates |
| GET | `/hnsw/graph?level=L&limit=N` | nodes + edges at a level (sampled for layer 0) with projected coords |
| GET | `/hnsw/stats` | nodes per level, avg/min/max degree per level, entry point, max level |
| POST | `/sweep` | body: `{param: nprobe\|efSearch, values: [...], k, n_queries}` → recall@k, mean/p95 latency per value |
| GET | `/pq/error` | reconstruction error distribution (histogram), per-list mean error, worst-50 ids |
| GET | `/metadata/{id}` | metadata row |
| GET | `/health` | ok |

Everything else (`/`, `/assets/*`) serves the built frontend.

---

## 8. Phases & steps

### Phase 0: Scaffold & reserve the name
1. `uv init --lib faissight` with `src/` layout; configure `pyproject.toml` (hatchling, metadata, extras from section 4, `[project.scripts] faissight = "faissight.cli:app"`).
2. Add ruff, mypy, pytest config; `.pre-commit-config.yaml`.
3. Scaffold `frontend/` with Vite React-TS + Tailwind; set Vite `build.outDir` to `../src/faissight/static` and `emptyOutDir: true`; add `static/` to `.gitignore` but include it in wheel via hatch `artifacts`.
4. Minimal `cli.py` with `faissight --version`.
5. CI workflow: Python 3.10/3.11/3.12 matrix, `uv sync --all-extras`, ruff, mypy, pytest; separate job builds the frontend.
6. **Manual step for the human** (note it in README TODO, do not attempt): publish a `0.0.1` placeholder to PyPI to reserve the name, and set up PyPI trusted publishing for `release.yml`.

**Acceptance:** `uv run faissight --version` works; CI green; `uv build` produces a wheel that contains `faissight/static/index.html`.

### Phase 1: Test fixtures & loader
1. `examples/make_synthetic.py`: generate `n=20_000, d=64` gaussian blobs (seeded, 50 centres) and save indexes: `IndexFlatL2`, `IndexIVFFlat(nlist=128)`, `IndexIVFPQ(nlist=128, m=8, nbits=8)`, `IndexIVFScalarQuantizer(QT_8bit)`, `IndexHNSWFlat(M=16)`, `IndexIDMap(IndexFlatL2)`, an `IndexPreTransform(PCA→IVFFlat)`, plus an IP-metric IVFFlat on normalised vectors. Also save the raw vectors `.npy` and a fake `chunks.jsonl`.
2. `tests/conftest.py`: session-scoped fixtures building **small** versions (n≈2,000) of each in `tmp_path_factory` so tests are fast.
3. `core/loader.py`:
   - `load_index(path_or_index) -> LoadedIndex` using `faiss.read_index`.
   - Unwrap chain: walk `IndexPreTransform`, `IndexIDMap`/`IndexIDMap2`, `IndexRefine` using `faiss.downcast_index`; record the class chain for display.
   - Use `faiss.extract_index_ivf(index)` (in a try/except) to find an IVF core.
   - Detect `IndexKind`: `FLAT`, `IVF_FLAT`, `IVF_PQ`, `IVF_SQ`, `HNSW_FLAT`, `HNSW_OTHER`, `UNSUPPORTED`.
   - Read params: `d`, `ntotal`, `metric_type`, `nlist`, `nprobe`, `pq.M`, `pq.nbits`, `hnsw.efSearch`, `hnsw.efConstruction`, M.
   - Id mapping: if IDMap, `faiss.vector_to_array(index.id_map)`.
4. `faissight info my.index` prints a rich table of the above.

**Acceptance:** loader tests pass for every synthetic index type; `info` output is correct and readable; unsupported types (e.g. a binary index) produce a clear message rather than a crash.

### Phase 2: IVF inspection
1. `core/ivf.py` using `faiss.contrib.inspect_tools` (`get_invlist_sizes`, `get_invlist`):
   - `list_sizes() -> np.ndarray`
   - `imbalance_factor()` (FAISS formula: `nlist * Σ size² / (Σ size)²`); cross-check against `invlists.imbalance_factor()` if available.
   - `list_members(list_no) -> ids`
   - `centroids()` via `ivf.quantizer.reconstruct_n(0, nlist)`
   - `assignments() -> ids→list_no` (build once, cache).
2. Map internal ids to user-facing ids when wrapped in IDMap.

**Acceptance:** Σ list sizes == ntotal; every id appears in exactly one list; centroids shape `(nlist, d)`; tests cover IVFFlat, IVFPQ, IVFSQ, and PreTransform-wrapped IVF (centroids live in the transformed space, so document that in DECISIONS.md).

### Phase 3: Search, ground truth, IVF trace
1. `core/search.py`:
   - `search(query, k, nprobe=None, efSearch=None)` using `faiss.SearchParametersIVF` / `faiss.SearchParametersHNSW` (don't mutate the shared index params; fall back to `faiss.ParameterSpace().set_index_parameter` only if params objects are unsupported by the index type).
   - `ground_truth(query, k)`: exact `IndexFlat` with the **same metric** built over raw vectors (or reconstructed vectors, with the flag set). Build lazily, cache.
   - `recall_at_k(results, truth)`.
   - Query resolution: by stored id (reconstruct, then exclude self from results), by raw vector, or by text via embedder.
2. IVF trace:
   - Coarse search: `quantizer.search(q, nlist)` → full probe order with centroid distances (apply the PreTransform to the query first if present).
   - Mark first `nprobe` as probed.
   - For each ground-truth neighbour: its list, the **probe rank** of that list, and a miss reason:
     - `CELL_NOT_PROBED` (needed nprobe ≥ rank+1)
     - `QUANTIZATION` (its cell was probed but approximate distance pushed it out of top-k; applies to PQ/SQ)
     - `FOUND`
   - Return "minimum nprobe to find all true top-k" as a headline number.
3. `faissight.core` public API summary in `__init__` of core.

**Acceptance:** on IVFFlat with nprobe=nlist, recall@10 == 1.0 against ground truth; miss reasons are correct on a hand-constructed test where a neighbour is placed in a far cell; self-exclusion works.

### Phase 4: Projection
1. `core/projection.py`:
   - PCA via numpy SVD (no sklearn dependency) to 2D/3D.
   - UMAP if `umap-learn` installed (`n_neighbors=15, min_dist=0.1, random_state=42`).
   - Fit on a sample (≤ 50k), project centroids and query vectors with the same fitted transform (for UMAP use `.transform`).
   - Stratified sampling by IVF list so small lists still appear.
   - Disk cache keyed by sha1 of index bytes + method + dims + sample size.
2. Background computation: projection runs in a thread when the server starts; API returns `202 {status: "computing"}` until ready; UI shows progress.

**Acceptance:** PCA of 20k×64 in < 2 s; cache hit on second run; query vector lands near its neighbours in projection space (sanity test: nearest projected point overlaps top-10 more than random).

### Phase 5: Server & CLI serve
1. `server/app.py` app factory taking a `Session`; mount `static/` with SPA fallback to `index.html`.
2. Implement endpoints from section 7 that are backed by phases 1–4 (`info`, `ivf/*`, `projection`, `search`, `trace/ivf`, `metadata`, `health`).
3. `faissight serve INDEX [--vectors v.npy] [--ids ids.npy] [--meta chunks.jsonl] [--embedder all-MiniLM-L6-v2] [--queries q.npy] [--host 127.0.0.1] [--port 8765] [--no-browser] [--max-points 50000]`. Opens the browser unless `--no-browser`.
4. Validate inputs up front with actionable errors (dimension mismatch between vectors and index, metadata missing `id` column, etc.).
5. `tests/test_api.py` using FastAPI `TestClient`.

**Acceptance:** `faissight serve examples/data/ivfflat.index --vectors ... --meta ...` starts, `/api/info` and `/api/search` return correct data; API tests pass.

### Phase 6: Frontend MVP
Layout: left sidebar (index info card + nav), main panel (active view). Light and dark theme.

1. **Overview view**: index card (kind, class chain, d, ntotal, metric, params), list-size histogram, imbalance factor with a plain-English verdict ("healthy" / "skewed: top 5% of lists hold X% of vectors"), empty-list count, top-20 largest lists table.
2. **Cluster Map view**: deck.gl scatter of sampled points coloured by list; centroids as larger outlined markers; hover shows id + metadata snippet; click a centroid → side panel with that list's members. Toggle 2D/3D, PCA/UMAP.
3. **Query Explorer view** (the core feature):
   - Query input: text (if embedder), stored id, or paste vector.
   - Controls: k, nprobe (or efSearch), "compare with exact" toggle.
   - Results table: rank, id, distance, list, metadata text, badge if also in ground truth.
   - Ground-truth table with miss reasons and colour-coded badges.
   - Headline: recall@k, min nprobe needed.
   - On the map: query star, probed cells highlighted, returned results vs missed true neighbours in different colours.
   - Probe-order chart: centroid distance vs probe rank, with the nprobe cut-off line and dots for cells holding true neighbours.
4. Loading/computing states, empty states, error toasts from structured errors.

**Acceptance:** full flow works on the synthetic IVFFlat and IVFPQ indexes; UI stays responsive with 50k points; Lighthouse performance reasonable on the built bundle.

### Phase 7: Recall / latency tuner
1. `core/sweep.py`: for each value in the sweep (default nprobe `[1,2,4,8,16,32,64,128,…≤nlist]`, efSearch `[16,32,64,128,256,512]`) run the query set, measure recall@k against cached ground truth and latency (warm-up run first; `time.perf_counter`; report mean and p95 per query; pin `faiss.omp_set_num_threads(1)` during timing and restore afterwards so numbers are stable and comparable).
2. Endpoint `/sweep` runs in a background thread with progress polling.
3. **Tuner view**: recall@k vs param line chart, latency vs param chart, and a recall-vs-latency Pareto chart; "target recall" input highlights the cheapest value meeting it; a copyable code snippet: `index.nprobe = 24` / `faiss.SearchParametersIVF(nprobe=24)`.
4. CLI: `faissight sweep INDEX --vectors ... --queries ... --param nprobe` prints a rich table (useful without the UI and in CI).

**Acceptance:** recall is monotonic non-decreasing in nprobe on IVFFlat (test); recommended value meets target recall on the test set.

### Phase 8: Python API & Jupyter
1. `faissight.launch(index, vectors=None, ids=None, metadata=None, embedder=None, port=None, open_browser=None) -> Viewer`:
   - Accepts in-memory `faiss.Index`, `np.ndarray`, pandas DataFrame or list of dicts for metadata, a callable embedder.
   - Starts uvicorn in a daemon thread on a free port.
   - In Jupyter: returns an object whose `_repr_html_` renders an `IFrame` of the UI (height 800). Outside Jupyter: opens the browser.
   - `Viewer.stop()` shuts down the server; `Viewer.url` property.
2. `examples/notebook_demo.ipynb`: build an index, launch, run a query.
3. Note in docs: works in JupyterLab and VS Code notebooks locally; Colab needs `google.colab.output.serve_kernel_port_as_iframe`; add a small helper for that and detect Colab automatically.

**Acceptance:** launching twice in one kernel uses different ports and doesn't crash; `stop()` frees the port; notebook demo runs top to bottom.

### Phase 9: HNSW view & search trace
1. `core/hnsw.py` (via `faiss.contrib.inspect_tools.get_hnsw_links` or direct `hnsw.levels/offsets/neighbors` arrays with `faiss.vector_to_array`):
   - node level for each id (FAISS stores `levels` as level+1).
   - neighbours per node per level; entry point; max level.
   - stats: nodes per level, degree distribution per level.
2. `core/hnsw_trace.py`: re-implement FAISS HNSW search in numpy over the extracted graph using stored vectors (reconstructed from HNSW storage):
   - Upper levels: greedy descent (ef=1) from entry point.
   - Level 0: beam search with `efSearch`, candidate min-heap + result max-heap.
   - Record visit order, distance at each visit, edges traversed, per level.
   - Note FAISS internals (e.g. `search_bounded_queue`, visited-table details) may differ slightly; this is a **reconstructed trace**. Label it so in the UI.
3. Validation test: trace results must overlap FAISS's own `search` results by ≥ 95% on average across 100 queries at efSearch=64. If lower, investigate and document differences in DECISIONS.md.
4. **HNSW Graph view**:
   - Stacked layer view (deck.gl OrbitView): each level a plane, top levels full, layer 0 sampled around the query neighbourhood.
   - Stats panel: nodes/level, degree histograms, entry point.
   - "Play search" animation: step through the trace (play/pause/step/speed), highlighting current node, frontier, visited set, and final top-k; drop from layer to layer.
   - Missed ground-truth neighbours highlighted with "not reached from the visited frontier" explanation.
5. efSearch support in the Query Explorer and Tuner for HNSW indexes.

**Acceptance:** trace overlap test passes; animation runs smoothly for efSearch up to 256; view works on the synthetic HNSWFlat index.

### Phase 10: Quantization (PQ / SQ) view
1. `core/pq.py`:
   - Requires raw vectors; otherwise the view shows "Provide --vectors to measure quantization error."
   - `make_direct_map()` on the IVF, reconstruct stored vectors in batches, compute per-vector error `‖x − x̂‖²` (and relative error).
   - Aggregate: histogram, per-list mean error, worst-50 ids.
   - Distance distortion: for sampled pairs, true vs approximate distance scatter.
2. **Quantization view**: error histogram, per-list error bar chart (sortable), true-vs-approx distance scatter with y=x line, worst vectors table with metadata text.
3. In Query Explorer, mark `QUANTIZATION` misses with a link to this view.

**Acceptance:** error is ~0 for IVFFlat, clearly > 0 for IVFPQ, lower for IVF-SQ8 than IVFPQ(m=8) in synthetic tests.

### Phase 11: RAG demo dataset & polish
1. `examples/make_rag_demo.py`:
   - Take ~20k passages from a public, permissively licensed text dataset (e.g. from Hugging Face `datasets`; choose one and record the licence in the README).
   - Chunk (≈200 tokens), embed with `sentence-transformers/all-MiniLM-L6-v2` (normalised, IP metric).
   - Build and save `IVFFlat(nlist=256)`, `IVFPQ(nlist=256, m=48, nbits=8)`, `HNSWFlat(M=32)`, plus `vectors.npy`, `chunks.jsonl`, and ~200 sample queries.
2. UI polish: keyboard shortcut to focus query box, shareable URL state (query + params in URL hash), export results as JSON/CSV, screenshot-friendly layout.
3. Performance pass: profile `/projection` and `/search` on 1M-vector synthetic index; ensure server memory stays reasonable (document limits in README).

**Acceptance:** a newcomer can go from `pip install` to seeing the RAG demo in under 2 minutes following the README.

### Phase 12: Docs, demo deployment, release 0.1.0
1. README:
   - One-line pitch, animated GIF of Query Explorer + HNSW animation, install, 3-command quick start, Python/Jupyter usage, supported index types table, "How it compares" section (Feder, embedding projectors, honest and short), limitations, roadmap, contributing.
2. `deploy/hf-space/Dockerfile`: Python slim, install `faissight[all]` from the repo, bake in the RAG demo artifacts, `CMD faissight serve ... --host 0.0.0.0 --port 7860 --no-browser`. Include a read-only "demo mode" flag that disables uploads and heavy sweeps beyond a query cap.
3. `release.yml`: on tag `v*`, build frontend, `uv build`, publish via PyPI trusted publishing, create GitHub release with changelog.
4. Tag `v0.1.0`.

**Acceptance:** `pip install "faissight[all]"` in a clean venv + README quick start works; HF Space loads and runs a query; PyPI page renders README correctly.

### Phase 13 (optional, after 0.1.0): VS Code extension wrapper
1. Separate folder `vscode-extension/` (TypeScript, `yo code` scaffold).
2. Command + explorer context menu on `*.index` / `*.faiss` files: "Open in faissight".
3. Uses the Python interpreter from the VS Code Python extension, checks `faissight` is installed (offer `pip install` if not), spawns `faissight serve FILE --no-browser --port <free>`, and shows the UI in a webview panel.
4. Kill the process when the panel closes.
5. Publish to VS Code Marketplace and Open VSX.

---

## 9. Testing strategy

- **Unit** (`tests/test_*.py`): every core module on tiny synthetic indexes; tests must run in < 60 s total.
- **Correctness anchors**: recall=1.0 at nprobe=nlist; Σ list sizes = ntotal; HNSW trace overlap ≥ 95%; PQ error ordering (Flat < SQ8 < PQ).
- **API**: `TestClient` covering each endpoint, including error cases (bad id, dimension mismatch, unsupported index).
- **Frontend**: Vitest for API client and pure helpers; one Playwright smoke test (load app → run query → results visible) in CI against the synthetic index.
- **Coverage target**: ≥ 85% on `faissight.core`.

---

## 10. Performance budgets

| Operation | Target (20k × 384, laptop CPU) |
|---|---|
| `faissight serve` to UI visible | < 3 s (projection may still be computing) |
| PCA projection | < 2 s |
| UMAP projection | < 60 s, cached afterwards |
| Single query with IVF trace + ground truth | < 150 ms |
| Sweep, 200 queries × 8 nprobe values | < 20 s |
| Frontend scatter, 50k points | ≥ 30 fps pan/zoom |

For 1M+ vectors: sampling kicks in, ground-truth index is built lazily, and the README documents RAM needs (raw vectors + flat ground truth ≈ 2 × n × d × 4 bytes).

---

## 11. Risks & mitigations

| Risk | Mitigation |
|---|---|
| FAISS SWIG API differences across versions | Pin minimum `faiss-cpu>=1.8`; use `faiss.contrib.inspect_tools` where possible; CI tests against latest and the minimum version |
| HNSW trace doesn't match FAISS exactly | Label as reconstructed; enforce overlap test; document differences |
| Huge indexes blow up memory | Sampling, lazy ground truth, `--max-points`, clear RAM guidance |
| UMAP is slow / heavy | PCA default, UMAP optional extra, disk cache |
| PreTransform indexes (OPQ/PCA) confuse the geometry | Always apply the transform to queries; show "transformed space" note in UI |
| Name collision later | Reserve on PyPI in Phase 0 |

---

## 12. Roadmap after 0.1.0 (put in README)

- IVF-HNSW (HNSW coarse quantizer) and IndexIVFFastScan support.
- Compare two indexes side by side (e.g. IVFFlat vs IVFPQ on the same queries).
- Evaluation with labelled query→relevant-chunk pairs (true retrieval relevance, not just ANN recall).
- Integrations: load a FAISS store directly from LangChain / LlamaIndex save folders.
- VS Code extension (Phase 13) if not already done.
