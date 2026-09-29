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

For very large indexes, lower `--max-points` to keep the browser light. Projections are cached on disk under `~/.cache/faissight/`, so restarts are fast.
