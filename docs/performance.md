# Performance and memory

Measured on an Apple M-series laptop (CPU only) with faissight 0.1 and faiss-cpu 1.15.

## 1M vectors (synthetic, 1,000,000 × 64, `IVF1024,Flat`)

| Operation | Time |
|---|---|
| `faissight serve` until the UI answers | 0.7 s |
| PCA projection ready (50k-point stratified sample) | 0.3 s |
| `GET /api/projection` (50k points, 1.3 MB JSON) | 10 ms |
| `POST /api/search` with exact ground truth and IVF trace | 7–9 ms |
| `POST /api/trace/ivf` | 9 ms |
| Sweep: 200 queries × 11 nprobe values | 2.7 s |

## Budgets from the plan (20k × 384, laptop CPU)

| Operation | Budget | Measured |
|---|---|---|
| Serve until UI visible | < 3 s | ~1 s |
| PCA projection | < 2 s | 0.05 s |
| UMAP projection | < 60 s | ~23 s, then cached |
| Query with IVF trace and ground truth | < 150 ms | < 10 ms |
| Sweep, 200 queries × 8 nprobe values | < 20 s | 0.24 s |
| Scatter with 50k points | ≥ 30 fps | ~60 fps |

## Memory

```
RAM ≈ index size + n × d × 4 bytes + ~100 MB
```

- **With `--vectors`:** the raw vectors are the n × d × 4 term. faissight doesn't copy them when their ids are already in order, and exact search runs directly over them with `faiss.knn` (no second copy).
- **Without `--vectors`:** the same term comes from vectors reconstructed from the index, the first time ground truth or a projection needs them.
- **Projections** hold at most `--max-points` points (50k by default), so they cost a few MB whatever the index size.

Measured at 1M × 64 (index 264 MB, vectors 256 MB): about 635 MB resident after projection, searches and a sweep, with or without `--vectors`.

Rough guide:

| Vectors | d = 384 (MiniLM) | d = 768 | d = 1536 |
|---|---|---|---|
| 100k | ~0.4 GB | ~0.7 GB | ~1.3 GB |
| 1M | ~3 GB | ~6 GB | ~12 GB |

(Flat/IVFFlat index + vectors; a PQ index is much smaller, so roughly halve these.)

### Memory-mapped vectors (`--mmap`)

`--mmap` (or `launch(..., mmap=True)`) maps the `--vectors` file instead of reading it into
RAM. Exact search, sweeps, comparisons and quantization analysis read the mapped file
directly. The pages still show up in RSS, but they are file-backed and clean, so the OS can
drop them under memory pressure instead of swapping. faissight keeps the mapping only if the
file holds C-ordered float32 rows sorted by id. Otherwise it copies the vectors into memory
and says why, both at startup and in the Overview.

Measured on an M-series Mac with 1M × 64 vectors (256 MB) and an IVF1024,PQ16 index
(24 MB): load, one exact search, then a 200-query sweep over two nprobe values.
"Footprint" is macOS's physical footprint, which excludes clean file-backed pages.

| Vectors | Mode | Mapped | Peak RSS | Footprint after | Peak footprint |
|---|---|---|---|---|---|
| ids 0..n-1 | in RAM | – | 401 MB | 338 MB | 362 MB |
| ids 0..n-1 | `--mmap` | yes | 402 MB | 94 MB | 118 MB |
| custom ids, unsorted rows | in RAM | – | 627 MB | 338 MB | 588 MB |
| custom ids, unsorted rows | `--mmap` | no (reordered) | 627 MB | 338 MB | 361 MB |

Custom ids in arbitrary row order cost a second copy while the rows are sorted by id: the
peak is loaded array + sorted copy. `--mmap` removes the first of those, cutting the peak
by about the size of the vectors, but the sorted copy stays in RAM. Save vectors and ids
sorted by id to map them fully.

Decoding the index (every vector as float32, plus int64 ids) is the other large cost. It is
needed for quantization analysis, and for ground truth and maps when `--vectors` is not
given. The Overview's Inputs card shows the estimate before anything is decoded, and the
Quantization view's progress message repeats it.

### HNSW inspection

The HNSW view reads the graph's link arrays and the stored vectors straight from the
index: flat storage is viewed in place, and SQ/PQ storage is decoded only for the nodes a
trace or drawn level touches. Degree statistics scan the level in bounded batches, and only
drawn nodes are projected. Measured with `benchmarks/hnsw_memory.py` (1M × 64,
`IndexHNSWFlat` M=16, vectors memory-mapped; extra peak physical footprint of one
operation after loading):

| Operation | Before | After |
|---|---|---|
| `GET /api/hnsw/stats` (degree statistics, all levels) | 549 MB | 54 MB |
| `GET /api/hnsw/graph` (2,000 nodes on level 0) | 639 MB | 61 MB |
| `POST /api/trace/hnsw` (efSearch 64) | 386 MB | 61 MB |

The PCA layout is fitted on the projection's 50k-point sample. Without `--vectors`, that
sample comes from decoding the index, which costs the n × d × 4 term above.

For very large indexes, lower `--max-points` to keep the browser light. Projections are cached on disk under `~/.cache/faissight/`, so restarts are fast.
