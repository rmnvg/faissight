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
| **Tuner** | Recall@k and latency across nprobe or efSearch, the cheapest value for your target recall (optionally with 95% confidence), per-query recall and the worst queries (one click to explain each), and a copy-paste snippet to apply it. |
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
                      [--compare other.index ...]
faissight info INDEX                          # kind, wrappers, parameters
faissight sweep INDEX --vectors v.npy [--param nprobe] [--target 0.95]
                      [--repeats 3] [--seed 0] [--json]
faissight compare LEFT RIGHT --vectors v.npy [--queries q.npy] [--ids ids.npy]
                      [--left-nprobe 4] [--right-ef-search 64] [--json]
faissight demo [--index ivf_pq|ivf_flat|hnsw]
```

`faissight sweep` exits with code 2 when no value reaches `--target`, so it can guard recall
in CI.

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
fastest measured setting, since that gap is usually timing noise.

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
