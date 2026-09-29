# Changelog

All notable changes to faissight are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[semantic versioning](https://semver.org/).

## [Unreleased]

### Added

- `faissight compare` and `core.compare_indexes`: paired recall/latency/size comparison
  with shared raw ground truth and per-query neighbour changes.
- Configurable sweep timing repeats and seeds, plus query fingerprints and environment
  metadata in JSON exports.
- Sweep cancellation in the Tuner and API, bounded background workers/queue, and bounded
  completed-job and query/ground-truth caches.
- Release gates for tagged-commit CI and isolated installed-wheel smoke tests.

### Fixed

- Reject raw-vector ID sets that disagree with the index instead of reporting false recall.
- Preserve large int64 IDs across API responses, searches, maps, HNSW traces, links and exports.
- Reject non-finite vectors/queries and fractional raw-vector IDs at ingestion.
- Display failed sweep-start and cancellation requests in the Tuner.

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
