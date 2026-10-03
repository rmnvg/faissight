# Changelog

All notable changes to faissight are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[semantic versioning](https://semver.org/).

## [Unreleased]

## [0.1.0] - 2026-10-03

First public release.

### Added

- **Getting started:** `faissight serve INDEX` opens the web UI; `faissight demo` builds a
  RAG demo (12,000 Apache-2.0 passages, MiniLM embeddings, IVF-Flat, IVF-PQ and HNSW) and
  opens it; `faissight.launch(...)` does the same from Python and Jupyter (JupyterLab,
  VS Code, Colab and JupyterHub proxies). Python 3.10 to 3.13.
- **Index support:** Flat, IVF-Flat, IVF-PQ, IVF-SQ and HNSW (Flat/SQ/PQ storage), through
  IDMap/IDMap2, PreTransform (PCA, OPQ) and Refine wrappers. Binary, GPU and FastScan
  indexes load with basic stats. User ids are preserved across the full int64 range in the
  API, links and exports.
- **Query explorer:** search by text, stored id, held-out evaluation query or vector against
  exact ground truth, with a reason for every missed neighbour (cell not probed,
  quantization, transform, ranked out), the smallest nprobe that finds them all, a
  probe-order chart, an optional exact-rerank experiment, shareable URLs and CSV/JSON export.
- **Tuner** and `faissight sweep`: recall@k and latency across nprobe or efSearch, with 95%
  recall intervals, per-query recall and the worst queries (one click to explain each), the
  cheapest setting for a target recall and optional p95 latency budget, suggested next steps
  backed by the measurements (including IVF probe coverage), a "How far to trust this"
  checklist, a copyable snippet, and cancellation. Exit codes gate CI on the target.
- **Saved runs and baselines:** `sweep --save`, `sweep --baseline`, `faissight runs diff`
  and the Tuner's **Save run** / **Compare with a saved run** keep a sweep as validated JSON
  and flag recall drops and p95 growth per setting, or say plainly when two runs aren't
  comparable (different queries, metric or corpus).
- **Compare** view and `faissight compare`: indexes over the same vectors measured on
  identical queries and ground truth: recall with intervals, mean/p95 latency, serialized
  size and per-query neighbour changes.
- **Relevance** view and `faissight evaluate`: labelled judgements (JSONL) give recall, MRR
  and nDCG@k before and after exact reranking, per query, with CSV/JSON export.
- **Run history:** finished sweeps, comparisons and evaluations are kept in
  `~/.cache/faissight/history/` (newest 100) across restarts; rename, download or diff them.
- **HNSW graph:** a layered 3D view and an animated, reconstructed search trace that matches
  FAISS's own results.
- **Quantization:** reconstruction error, compression ratio and near-pair distance fidelity
  for PQ/SQ codes.
- **Overview** (list sizes, imbalance verdict, largest lists, a "Diagnose my index" guide)
  and **Cluster map** (PCA/UMAP in 2D/3D, up to 50,000 stratified points, click a list to
  read its chunks).
- **Large indexes:** `--mmap` memory-maps raw vectors; the Overview reports memory costs.
  HNSW inspection doesn't copy the graph or vectors, background work is bounded and
  cancellable, and projections are cached on disk (`faissight cache info | clear`).
- **Public demos:** a read-only `--demo-mode` with capped workloads, and a Hugging Face
  Space image (`deploy/hf-space/`).
