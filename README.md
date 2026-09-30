# faissight

**See inside your FAISS index: debug retrieval, tune recall.**

[![CI](https://github.com/rmnvg/faissight/actions/workflows/ci.yml/badge.svg)](https://github.com/rmnvg/faissight/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/faissight)](https://pypi.org/project/faissight/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue)](https://github.com/rmnvg/faissight/blob/main/LICENSE)

FAISS is the default vector index in countless RAG systems, but it has no UI. When retrieval
goes wrong, faissight answers the questions you can't easily ask FAISS directly:

- Which cells did my query probe, and were the true nearest neighbours even in them?
- Why did this irrelevant chunk rank first, and where did the relevant one go?
- Is my IVF index badly imbalanced?
- How much accuracy did PQ compression cost me?
- Which `nprobe` / `efSearch` gives 95% recall at the lowest latency?

![Query explorer: recall 0.40 at nprobe 4, five neighbours missed because their cells weren't probed, one lost to quantization; nprobe 25 would find them all](https://raw.githubusercontent.com/rmnvg/faissight/main/docs/screenshots/query-explorer.png)

## Quick start

```bash
pip install "faissight[all]"
faissight demo                       # builds a real RAG demo once (~1 min), opens the UI
faissight serve my.index --vectors vectors.npy --meta chunks.jsonl   # your own index
```

`faissight demo` downloads 12,000 public passages, embeds them with `all-MiniLM-L6-v2` and
builds IVF-Flat, IVF-PQ and HNSW indexes. Type a question and see which true neighbours
the index missed, and why.

Only need FAISS support? `pip install "faissight[faiss-cpu]"` is enough; `[all]` adds text
queries (sentence-transformers), UMAP and Parquet metadata. Conda or GPU users with FAISS
already installed can `pip install faissight`.

## From Python and Jupyter

```python
import faissight

viewer = faissight.launch(index, vectors=xb, metadata=chunks)  # any faiss.Index
viewer  # in a notebook: the UI appears inline; elsewhere your browser opens
viewer.stop()
```

`metadata` can be a list of dicts, a DataFrame or a `.jsonl`/`.csv`/`.parquet` file with an
`id` column. `embedder` can be any `str -> vector` function. It works in JupyterLab, VS Code
notebooks and Colab (detected automatically); see [Jupyter notes](https://github.com/rmnvg/faissight/blob/main/docs/jupyter.md).

The same analyses are available without the UI:

```python
report = viewer.session.query(id=42, k=10, nprobe=4)
report.recall, report.ivf_trace.min_nprobe_for_all
```

## What's inside

| View | What it shows |
|---|---|
| **Query explorer** | Search by text, stored id or vector. Compares with exact ground truth and gives a reason for every missed neighbour: *cell not probed*, *quantization*, *transform* (PCA/OPQ), or *ranked out*, plus the smallest nprobe that finds them all. Shareable URLs, CSV/JSON export. |
| **Tuner** | Recall@k and latency across nprobe or efSearch, the cheapest value for your target recall (optionally with 95% confidence), per-query recall and the worst queries (one click to explain each), suggested next steps backed by the measurements, and a copy-paste snippet to apply it. |
| **Compare** | Two indexes over the same vectors (say IVF-Flat vs IVF-PQ vs HNSW) on identical queries: recall with intervals, mean/p95 latency, serialized size, and which queries' neighbours changed. Start with `--compare OTHER.index`. |
| **HNSW graph** | The layers of an HNSW graph and an animated, step-by-step replay of the search, showing which true neighbours were never reached. |
| **Quantization** | How far PQ/SQ codes are from your raw vectors, the compression ratio, and how much they distort distances between near neighbours (the part that decides ranking). |
| **Overview** | Inverted-list sizes, imbalance factor with a plain-English verdict, largest lists. |
| **Cluster map** | Stored vectors projected with PCA or UMAP (2D/3D), with IVF centroids; click a list to read its chunks. |

![Animated HNSW search: greedy descent through the upper layers, then candidate expansion on level 0](https://raw.githubusercontent.com/rmnvg/faissight/main/docs/screenshots/hnsw-search.png)

<table>
<tr>
<td><img alt="Tuner: recall and latency by nprobe with a recommendation" src="https://raw.githubusercontent.com/rmnvg/faissight/main/docs/screenshots/tuner.png"></td>
<td><img alt="Quantization: PQ error, compression and distance fidelity" src="https://raw.githubusercontent.com/rmnvg/faissight/main/docs/screenshots/quantization.png"></td>
</tr>
<tr>
<td><img alt="Cluster map with one inverted list selected" src="https://raw.githubusercontent.com/rmnvg/faissight/main/docs/screenshots/cluster-map.png"></td>
<td><img alt="Overview: list balance and sizes" src="https://raw.githubusercontent.com/rmnvg/faissight/main/docs/screenshots/overview.png"></td>
</tr>
</table>

## Inputs

| Input | Required | Formats | Used for |
|---|---|---|---|
| FAISS index | yes | index file or in-memory `faiss.Index` | everything |
| Raw vectors | recommended | `.npy` float32, n × d (row *i* = id *i*, or use `--ids`) | exact ground truth, quantization error |
| Ids | optional | `.npy` int64 | when row order isn't the id |
| Metadata | optional | `.jsonl` / `.csv` / `.parquet` with an `id` column (plus e.g. `text`, `title`) | reading results as text |
| Embedder | optional | sentence-transformers model name, or a Python callable | text queries |
| Query vectors | optional | `.npy` | sweeps on your own queries |

Without raw vectors, faissight reconstructs them from the index; the UI then warns that
ground truth is computed on reconstructed vectors, so PQ/SQ error isn't measured.

## Supported indexes

| Index | Support |
|---|---|
| `IndexFlat` (L2 / IP) | search, map, ground truth |
| `IndexIVFFlat`, `IndexIVFPQ`, `IndexIVFScalarQuantizer` | everything: list balance, probe trace, miss reasons, tuner, quantization |
| `IndexHNSWFlat`, `IndexHNSWSQ`/`PQ` | graph view, search trace, tuner (efSearch), quantization for SQ/PQ storage |
| Wrappers: `IndexIDMap`/`IDMap2`, `IndexPreTransform` (PCA, OPQ), `IndexRefine` | unwrapped transparently; ids stay yours |
| Binary, GPU, IVF FastScan, others | basic stats only (convert GPU indexes with `faiss.index_gpu_to_cpu`) |

## Command line

```text
faissight serve INDEX [--vectors v.npy] [--ids ids.npy] [--meta chunks.jsonl]
                      [--embedder all-MiniLM-L6-v2] [--queries q.npy]
                      [--host 127.0.0.1] [--port 8765] [--no-browser] [--max-points 50000]
                      [--compare other.index ...] [--mmap]
faissight info INDEX                          # kind, wrappers, parameters
faissight sweep INDEX --vectors v.npy [--param nprobe] [--target 0.95]
                      [--max-p95-ms 10] [--repeats 3] [--seed 0] [--json]
                      [--save run.json] [--baseline earlier.json]
faissight runs diff EARLIER.json LATER.json   # regressions between saved runs
faissight compare LEFT RIGHT --vectors v.npy [--queries q.npy] [--ids ids.npy]
                      [--left-nprobe 4] [--right-ef-search 64] [--json]
faissight demo [--index ivf_pq|ivf_flat|hnsw]
faissight cache info | clear [--older-than-days 30] [--max-mb 500]
```

`faissight sweep` exits with code 2 when no value reaches `--target` (within the p95 budget
of `--max-p95-ms`, if given), 3 when `--baseline` shows a regression, and 4 when `--baseline`
couldn't be compared at all (different query sets, no overlapping values, ...) &mdash; that is
not a pass, and CI should treat it as a failure, not silently skip the check. `faissight runs
diff` uses the same three failure codes.

### Saved runs and baselines

`faissight sweep --save run.json` (or **Save run** in the Tuner) keeps a sweep as JSON: the
index's sha1, parameters and distance metric, the query set's fingerprint, a fingerprint of
the raw `--vectors` corpus ground truth was computed on, the settings, the environment,
every measurement, the recommendation for your target, and the worst queries. Compare a
later sweep with it using `--baseline run.json`, the Tuner's **Compare with a saved run**, or
`faissight runs diff`. A setting regresses when its recall drops by more than
`--max-recall-drop` (0.01), or when its p95 grows by more than `--max-p95-increase` (20%)
*and* by more than `--min-p95-increase-ms` (0.05 ms): sub-millisecond timings vary between
runs, and the floor keeps that noise from failing CI.

Recall is compared only when both runs used the same k, parameter, query set, metric and
ground-truth corpus; latency only when both were measured with the same FAISS version and
machine. Otherwise the comparison says why, and reports itself as not comparable rather than
reporting a false "no regressions". A saved run is validated on load and before it's written:
every recall and latency number must be finite and in range, point values must be unique, and
the identifying fingerprints must be present, so a corrupted or hand-edited file is rejected
rather than silently compared against.

```bash
faissight sweep index.faiss --vectors v.npy --queries eval.npy --save baseline.json
# ... rebuild or retrain the index ...
faissight sweep new.faiss --vectors v.npy --queries eval.npy --baseline baseline.json
```

## Reproducible tuning and index comparison

Sweeps warm up each setting, then time each query three times on one FAISS thread.
Use `--repeats` and `--seed` (also available in the Tuner) to control the measurement.
JSON exports include the query-set SHA-256, sampling seed, timing seed, repetition count,
and Python/NumPy/FAISS and platform versions. Supplied `--queries` use the first
`--n-queries` rows; the seed controls sampling only when queries are drawn from stored vectors.

Each setting also reports an approximate 95% interval for mean recall, the exact per-query
recall distribution (so "how many queries fall below the target" is known, not just the
mean) and its lowest-recall queries. The recommendation is the *smallest* value meeting the
target; when a larger value happens to measure faster, it is reported separately as the
fastest measured setting, since that gap is usually timing noise. A p95 latency budget
(`--max-p95-ms`, or the Tuner's budget field) adds a second constraint. When the target is
reachable but not within the budget, the result says so, with the setting recall needs and
the best recall the budget allows, instead of reporting "not reached".

### Suggested next steps

The Tuner, `faissight sweep` and `GET /api/sweep/{job_id}/advice?target=0.95` turn a sweep
into next steps, each with the measurements behind it. For IVF sweeps, every setting also
reports the share of true neighbours in the probed lists. That share caps recall, so the gap
between them tells the two failure modes apart:

- **Neighbours in lists that weren't probed**: raise nprobe. The suggestion names the
  nprobe at which the target share of neighbours is within reach and offers that sweep.
- **Neighbours in probed lists but ranked out**: more probing won't help. PQ/SQ codes, a
  PCA transform, or (for IVF-Flat) vectors that differ from what was indexed are named as
  the cause, with a fix such as re-ranking with exact distances.

HNSW sweeps suggest larger efSearch values while recall still rises fast enough to reach the
target, and a denser graph when it has levelled off. At the recommended setting, a tail of
queries finding fewer than half their neighbours, and uneven IVF lists, are flagged too.

Sampled stored vectors make convenient queries, but they are not your users' queries. Pass
held-out queries with `--queries` before trusting a recommendation; the Tuner and Compare
views show a "How far to trust this" checklist. Recall is agreement with exact search on
the same embeddings. It is not answer quality: if the exact neighbours are poor answers, the
embedding model or chunking is at fault, not the index.

```bash
faissight compare exact.index compressed.index --vectors vectors.npy \
  --queries queries.npy --right-nprobe 16 --repeats 5 --seed 42 --json > comparison.json
```

Comparison uses the same raw ground truth and queries for both indexes, reports recall,
mean/p95 latency and serialized index size, and lists neighbours unique to either index
for each query. Serialized size is not process RAM. Both indexes must use the same metric,
input dimension and user IDs. Python users can call `faissight.core.compare_indexes` with
loaded indexes, a `VectorSource`, and a `QuerySet`.

To compare in the UI, pass the other indexes to `serve` (or `launch(..., compare=[...])`);
they must be built from the same `--vectors`:

```bash
faissight serve ivf_flat.index --vectors vectors.npy --compare ivf_pq.index --compare hnsw.index
```

The **Compare** view runs the same paired measurement as `faissight compare`, with each
index's own nprobe/efSearch, and links changed queries to the Query Explorer.

Raw-vector IDs are checked against index IDs before analysis. Supply `--ids` for custom
IDs. The API uses numeric IDs through `2**53 - 1` and decimal strings above that limit,
preserving the full non-negative int64 range in browser inputs, links and exports.

Background work uses at most two workers and eight queued jobs per session; a full queue
returns `429 JOB_CAPACITY`. The Tuner's **Cancel sweep** button (or
`DELETE /api/sweep/{job_id}`) stops queued work immediately and running work at its next
checkpoint. A native FAISS call already running must finish first. Sessions retain up to
32 completed jobs for one hour, with expired entries removed on the next access; query and
ground-truth caches are limited to four entries and 64 MiB. An evicted sweep can be rerun.

Projections are cached on disk under `~/.cache/faissight/<index sha1>/` (or
`$FAISSIGHT_CACHE_DIR`). The cache only saves time: if it can't be written (a read-only home
or a full disk), the projection is still shown, with a warning in the Cluster map and the
server log. `faissight cache info` shows usage per index. `faissight cache clear` removes
everything, or only entries unused for `--older-than-days`, or least recently used entries
beyond `--max-mb`. It never touches the demo data. `serve --no-cache` skips the disk cache.

## How it compares

- **[Feder](https://github.com/zilliztech/feder)** (Zilliz) visualises FAISS IVF-Flat and
  hnswlib indexes in JavaScript and is great for learning how they search. faissight does
  all FAISS work in Python, supports PQ/SQ, HNSW and wrapper indexes, and focuses on
  debugging: exact ground truth, per-neighbour miss reasons, recall/latency tuning and
  chunk text.
- **Embedding projectors** (TensorFlow Projector, Nomic Atlas, Renumics Spotlight, Arize
  Phoenix) visualise embeddings. faissight visualises the *index*: what it probes, stores and
  loses.

## Limitations

- CPU FAISS only; convert GPU indexes first.
- The HNSW trace is reconstructed in Python. It matches FAISS's own results in our tests
  (100% overlap across efSearch 4–256, L2 and IP), but FAISS internals can change.
- Memory is roughly index size + n × d × 4 bytes (about 0.6 GB for 1M × 64, measured);
  see [performance and memory](https://github.com/rmnvg/faissight/blob/main/docs/performance.md).
- A local, single-user tool: no authentication. Use `--demo-mode` for a read-only public
  deployment ([Hugging Face Space setup](https://github.com/rmnvg/faissight/blob/main/deploy/hf-space/README.md)).

## Roadmap

- IVF-HNSW (HNSW coarse quantizer) and IVF FastScan support
- Bring the CLI/Python index comparison into a side-by-side web view
- Evaluation with labelled query→relevant-chunk pairs (retrieval relevance, not just ANN recall)
- Load FAISS stores directly from LangChain / LlamaIndex save folders
- VS Code extension

## Contributing

```bash
git clone https://github.com/rmnvg/faissight && cd faissight
uv sync --all-extras
cd frontend && npm install && npm run build && cd ..
uv run pytest && uv run ruff check . && uv run mypy
uv run python examples/make_synthetic.py && uv run faissight serve examples/data/ivf_pq.index --vectors examples/data/vectors.npy
```

Frontend development: run `npm run dev` in `frontend/` next to a running `faissight serve`.
Design decisions are logged in [docs/DECISIONS.md](https://github.com/rmnvg/faissight/blob/main/docs/DECISIONS.md).

## License

MIT. The demo corpus is "RAG Dataset 12000" by Neural Bridge AI (Apache-2.0), with passages
from Falcon RefinedWeb (ODC-By 1.0).
