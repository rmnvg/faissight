# Changelog

All notable changes to faissight are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[semantic versioning](https://semver.org/).

## [Unreleased]

### Added

- **Relevance** view: load labelled judgements (JSONL, as for `faissight evaluate`) and see
  recall, MRR and nDCG@k before and after exact reranking, per-query scores with links to
  the Query Explorer, and CSV/JSON exports. API: `POST /api/evaluation`, then poll
  `GET /api/evaluation/{job_id}`; `DELETE` cancels.
- **Run history**: finished sweeps, comparisons and relevance evaluations are saved under
  `~/.cache/faissight/history/` (newest 100) and survive restarts. Rename, download or
  delete them, and compare two saved sweeps for regressions. API: `/api/history`.
- A weekly **Performance** workflow measures this commit against the latest release on the
  same runner (`benchmarks/regression.py`) and fails on the plan's budgets or a 2x slowdown.
- Browser tests also run on Firefox and WebKit, with keyboard checks for the narrow-screen
  menu and the Tuner's results table.

### Changed

- Searches and traces run on their own small worker pool and stop when the browser abandons
  the request (a newer query or a closed tab), instead of finishing in the background.
- The frontend's API types are generated from the server's OpenAPI schema; CI fails if
  they drift.
- PCA projections use about a third less peak memory (identical results).

### Fixed

- Tuner result rows can be opened with the keyboard (Enter or Space), and the narrow-screen
  menu keeps focus inside it, closes with Escape and returns focus to the Menu button.

## [0.2.0] - 2026-10-03

### Added

- Saved runs and baselines: `faissight sweep --save` and the Tuner's **Save run** keep a
  sweep as versioned JSON (index sha1, query fingerprint, settings, environment,
  measurements, decision, worst queries). `sweep --baseline`, **Compare with a saved run**
  and `faissight runs diff` flag recall drops and p95 growth per setting (exit code 3 in
  CI), and say when runs aren't comparable. API: `GET /api/sweep/{job_id}/run`,
  `POST /api/sweep/{job_id}/baseline`; library: `core.run_record`, `compare_runs`.
- `faissight cache info` and `faissight cache clear [--older-than-days N] [--max-mb M]`
  show and prune the projection disk cache, least recently used first (cache hits refresh
  an entry). Demo data beside the cache is left alone.
- p95 latency budgets: `faissight sweep --max-p95-ms`, a budget field in the Tuner and
  `max_p95_ms` on `/api/sweep/{job_id}/advice`. The recommendation must meet recall and the
  budget. When no setting does both, the result names the setting recall needs and the best
  recall within the budget (`SweepResult.choose`), and exit code 2 gates CI on both.
- Held-out evaluation queries (`--queries`) can be explained like stored ones: "Explain" in
  the Tuner's worst queries and Compare's changed queries opens the Query Explorer by the
  query's row (`#/query?row=N`), and the explorer has an "Evaluation query" mode. The API
  accepts `{"query": {"row": N}}` in `/api/search` and `/api/trace/hnsw`.
- Suggested next steps in the Tuner, `faissight sweep` and `GET /api/sweep/{job_id}/advice`:
  probe more lists, compression/PCA/ranking limits, larger or no more efSearch, failing
  queries behind a good mean, and uneven IVF lists, each with the measurements behind it
  and a one-click follow-up (a suggested sweep, or the view that shows more). IVF sweeps
  now report probe coverage, the share of true neighbours in probed lists, per setting and
  per worst query (`core.advise`, `SweepPoint.probe_coverage`).
- "How far to trust this" checklist in the Tuner and Compare views: whether queries are
  held-out or sampled stored vectors, whether ground truth is exact, how precisely recall is
  measured (and how many queries would tighten it), and that ANN recall is not relevance.
  `faissight sweep` says when its queries are sampled stored vectors.
- `--mmap` for `serve`, `sweep` and `compare` (and `launch(..., mmap=True)`) memory-maps the
  raw vectors file. `/api/info.memory` and the Overview report the vectors' size, whether
  they are mapped (and why not, if not), and what decoding the index would cost.
- `faissight compare` and `core.compare_indexes`: paired recall/latency/size comparison
  with shared raw ground truth and per-query neighbour changes.
- Configurable sweep timing repeats and seeds, plus query fingerprints and environment
  metadata in JSON exports.
- Sweep cancellation in the Tuner and API, bounded background workers/queue, and bounded
  completed-job and query/ground-truth caches.
- Release gates for tagged-commit CI and isolated installed-wheel smoke tests.
- Compare view: `faissight serve --compare OTHER.index` (repeatable) and
  `launch(..., compare=[...])` put other indexes over the same vectors side by side in the UI,
  with recall intervals, mean/p95 latency, serialized size and changed neighbours per query.
  Backed by `POST/GET/DELETE /api/compare` jobs.
- Tuner diagnostics: 95% recall intervals, exact per-query recall distributions, the worst
  queries at each setting (linked to the Query Explorer), an optional "95% lower bound meets
  the target" rule, and the fastest measured setting reported apart from the recommendation.
  `faissight sweep` prints the interval and the share of queries below the target.

### Fixed

- `faissight demo --max-chunks N` with N below 256 downloaded and embedded the corpus, then
  crashed with a FAISS traceback (IVF-PQ needs 256 training vectors). It now fails at once
  with a clear message.
- Saved-run comparisons that couldn't judge anything (different query sets, no shared
  settings, incompatible metrics) reported `regressed: false` and exited 0 like a clean
  pass. `RunComparison.comparable` (and the API's `comparable` field) now distinguishes
  "compared and clean" from "nothing was checked"; `sweep --baseline` and `runs diff` exit
  4 for the latter, and the CLI/Tuner say so plainly.
- Saved runs are now validated, not just shape-checked: NaN/Infinity (JSON's own or the
  non-standard `NaN`/`Infinity` tokens some encoders emit), out-of-range recall, negative
  latency, and duplicate point values are rejected on save and on load, instead of silently
  producing a wrong or empty comparison.
- A saved run's identity now covers the distance metric and (for raw ground truth) a
  fingerprint of the `--vectors` corpus, so comparing across a metric change or a swapped
  vectors file is flagged as not comparable instead of silently diffing unrelated numbers.
- `GET /api/sweep/{job_id}/run` returned bare JSON integers for stored ids, so a "Save run"
  download could silently round an id beyond 2**53-1 (JavaScript's exact integer range)
  before it ever reached disk. It now uses the same string encoding as every other
  endpoint that returns ids.
- `faissight sweep` built a run record (hashing the whole index file) even without `--save`
  or `--baseline`; it's now built only when one of them is given.
- Loading a malformed file into the Tuner's "Compare with a saved run" left the previous
  comparison on screen looking current; the old result now clears immediately. Changing the
  regression thresholds without re-running is now flagged so a stale verdict can't be
  mistaken for one that reflects the current inputs.
- A projection whose disk-cache write fails (read-only or full disk) is still returned,
  with `cache_warning` in `/api/projection`, a Cluster map banner and a server log line;
  before, the write error failed the whole projection.
- Cancelling a sweep or comparison takes effect within one batch of 256 queries. Exact
  ground truth, warm-up and probe coverage used to run over the whole query set in one
  native call (11 s for 5,000 queries over 1M vectors) before the job could stop.
- HNSW inspection no longer copies the graph or the stored vectors: stats, level views and
  traces on a 1M-vector index peak at about 60 MB extra instead of 390–640 MB. Only drawn
  nodes are projected, and SQ/PQ storage is decoded per node (`benchmarks/hnsw_memory.py`).
- Restarting a cancelled or failed sweep or comparison with the same settings now shows the
  new run; before, the reused job id kept the old status on screen and polling never resumed.
- Sweep and comparison links carry their settings, and a job the server no longer has (expired
  or server restarted) says so and offers "Run again" with those settings.
- A server restart during a running sweep or comparison no longer leaves a stale progress bar
  beside the "no longer on the server" notice.
- Empty indexes no longer crash exact search, IVF traces, HNSW traces or quantization
  analysis: searches return no results and analyses answer `400 EMPTY_INDEX` with a hint.
- Reject raw-vector ID sets that disagree with the index instead of reporting false recall.
- Preserve large int64 IDs across API responses, searches, maps, HNSW traces, links and exports.
- Reject non-finite vectors/queries and fractional raw-vector IDs at ingestion.
- Display failed sweep-start and cancellation requests in the Tuner.
- On narrow frames (phones, notebook iframes) the sidebar starts hidden and opens as an
  overlay instead of squeezing the page.
- Show a persistent error with "Try again" on the Overview when the list-size request fails,
  instead of an empty page after the toast disappears.

## [0.1.0] - 2026-09-29

First release.

### Added

- `faissight serve INDEX` web UI, `faissight info`, `faissight sweep`, `faissight demo`, and
  `faissight.launch(...)` for Python and Jupyter (Colab and JupyterHub proxies supported).
- Index support: Flat, IVF-Flat, IVF-PQ, IVF-SQ and HNSW, through IDMap/IDMap2,
  PreTransform (PCA, OPQ) and Refine wrappers. Binary, GPU and FastScan indexes load with
  basic stats only.
- **Overview:** list-size histogram, imbalance factor with a plain-English verdict, and the
  largest lists.
- **Cluster map:** PCA/UMAP in 2D/3D with stratified sampling (up to 50k points), list
  highlighting and member lists with chunk text.
- **Query explorer:** search by text, stored id or vector; exact ground truth; recall@k; a
  reason for every missed neighbour (cell not probed, quantization, transform, ranked out);
  the minimum nprobe needed; probe-order chart; shareable URLs; CSV/JSON export.
- **Tuner:** recall and latency sweeps over nprobe/efSearch with a target-recall
  recommendation and a copyable code snippet.
- **HNSW graph:** layered 3D view and an animated, reconstructed search trace that matches
  FAISS's own results.
- **Quantization:** reconstruction error, compression, and near-pair distance fidelity for
  PQ/SQ codes.
- RAG demo on an Apache-2.0 passage dataset, a Hugging Face Space image, and a
  read-only `--demo-mode`.
