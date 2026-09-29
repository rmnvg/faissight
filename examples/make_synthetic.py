"""Generate synthetic gaussian-blob FAISS indexes of every supported type.

Used for quick demos and (at a smaller size) by the test suite.

    uv run python examples/make_synthetic.py --out examples/data

Writes, into ``--out``:

- ``<name>.index`` for each index in :data:`INDEX_NAMES`
- ``vectors.npy`` (raw float32 vectors, row i <-> id i)
- ``vectors_normalized.npy`` (L2-normalised copy, used by the IP index)
- ``ids_idmap.npy`` (the user-facing ids of ``idmap_flat``, aligned with ``vectors.npy`` rows)
- ``queries.npy`` (held-out queries drawn from the same blobs)
- ``chunks.jsonl`` (fake chunk metadata keyed by id)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import faiss
import numpy as np

INDEX_NAMES = (
    "flat_l2",
    "ivf_flat",
    "ivf_pq",
    "ivf_sq8",
    "hnsw_flat",
    "idmap_flat",
    "pca_ivf_flat",
    "ivf_flat_ip",
)

_WORDS = [
    "vector",
    "index",
    "cluster",
    "centroid",
    "query",
    "recall",
    "latency",
    "probe",
    "graph",
    "layer",
    "neighbour",
    "distance",
    "embedding",
    "chunk",
    "passage",
    "retrieval",
    "quantizer",
    "residual",
]


def make_blobs(n: int, d: int, n_centers: int = 50, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(vectors, labels)``: ``n`` float32 points around ``n_centers`` gaussian centres."""
    rng = np.random.default_rng(seed)
    centers = rng.normal(scale=4.0, size=(n_centers, d)).astype(np.float32)
    labels = rng.integers(0, n_centers, size=n)
    x = centers[labels] + rng.normal(size=(n, d)).astype(np.float32)
    return x.astype(np.float32), labels


def _normalize(x: np.ndarray) -> np.ndarray:
    return (x / np.linalg.norm(x, axis=1, keepdims=True)).astype(np.float32)


def idmap_ids(n: int) -> np.ndarray:
    """User-facing ids for ``idmap_flat``: sparse and non-contiguous on purpose."""
    return (np.arange(n, dtype=np.int64) * 7 + 1_000).astype(np.int64)


def build_indexes(
    x: np.ndarray,
    *,
    nlist: int = 128,
    pq_m: int = 8,
    pq_nbits: int = 8,
    hnsw_m: int = 16,
) -> dict[str, faiss.Index]:
    """Build and populate one index of each synthetic type over ``x``."""
    n, d = x.shape
    xn = _normalize(x)
    out: dict[str, faiss.Index] = {}

    flat = faiss.IndexFlatL2(d)
    flat.add(x)
    out["flat_l2"] = flat

    ivf_flat = faiss.IndexIVFFlat(faiss.IndexFlatL2(d), d, nlist)
    ivf_flat.train(x)
    ivf_flat.add(x)
    out["ivf_flat"] = ivf_flat

    ivf_pq = faiss.IndexIVFPQ(faiss.IndexFlatL2(d), d, nlist, pq_m, pq_nbits)
    ivf_pq.train(x)
    ivf_pq.add(x)
    out["ivf_pq"] = ivf_pq

    ivf_sq = faiss.IndexIVFScalarQuantizer(
        faiss.IndexFlatL2(d), d, nlist, faiss.ScalarQuantizer.QT_8bit
    )
    ivf_sq.train(x)
    ivf_sq.add(x)
    out["ivf_sq8"] = ivf_sq

    hnsw = faiss.IndexHNSWFlat(d, hnsw_m)
    hnsw.add(x)
    out["hnsw_flat"] = hnsw

    idmap = faiss.IndexIDMap(faiss.IndexFlatL2(d))
    idmap.add_with_ids(x, idmap_ids(n))
    out["idmap_flat"] = idmap

    pca_ivf = faiss.index_factory(d, f"PCA{d // 2},IVF{nlist},Flat")
    pca_ivf.train(x)
    pca_ivf.add(x)
    out["pca_ivf_flat"] = pca_ivf

    ivf_ip = faiss.IndexIVFFlat(faiss.IndexFlatIP(d), d, nlist, faiss.METRIC_INNER_PRODUCT)
    ivf_ip.train(xn)
    ivf_ip.add(xn)
    out["ivf_flat_ip"] = ivf_ip

    return out


def write_chunks(path: Path, labels: np.ndarray, seed: int = 0) -> None:
    """Write fake RAG chunk metadata (``id``, ``text``, ``source``, ``title``) as JSONL."""
    rng = np.random.default_rng(seed)
    with path.open("w") as f:
        for i, label in enumerate(labels):
            words = " ".join(rng.choice(_WORDS, size=12))
            row = {
                "id": i,
                "text": f"Synthetic passage {i} from blob {label}: {words}.",
                "source": f"blob_{label:02d}.txt",
                "title": f"Blob {label} / chunk {i}",
            }
            f.write(json.dumps(row) + "\n")


def build_dataset(
    out_dir: Path,
    *,
    n: int = 20_000,
    d: int = 64,
    n_centers: int = 50,
    n_queries: int = 200,
    nlist: int = 128,
    pq_m: int = 8,
    pq_nbits: int = 8,
    hnsw_m: int = 16,
    seed: int = 0,
) -> dict[str, Path]:
    """Build every synthetic index plus side files into ``out_dir``; return name -> path."""
    out_dir.mkdir(parents=True, exist_ok=True)
    x_all, labels_all = make_blobs(n + n_queries, d, n_centers, seed)
    x, labels, queries = x_all[:n], labels_all[:n], x_all[n:]

    paths: dict[str, Path] = {}
    for name, index in build_indexes(
        x, nlist=nlist, pq_m=pq_m, pq_nbits=pq_nbits, hnsw_m=hnsw_m
    ).items():
        paths[name] = out_dir / f"{name}.index"
        faiss.write_index(index, str(paths[name]))

    side_files = {
        "vectors": x,
        "vectors_normalized": _normalize(x),
        "ids_idmap": idmap_ids(n),
        "queries": queries,
    }
    for name, arr in side_files.items():
        paths[name] = out_dir / f"{name}.npy"
        np.save(paths[name], arr)

    paths["chunks"] = out_dir / "chunks.jsonl"
    write_chunks(paths["chunks"], labels, seed)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=Path(__file__).parent / "data")
    parser.add_argument("--n", type=int, default=20_000)
    parser.add_argument("--d", type=int, default=64)
    parser.add_argument("--nlist", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    paths = build_dataset(args.out, n=args.n, d=args.d, nlist=args.nlist, seed=args.seed)
    for name, path in paths.items():
        print(f"{name:20s} {path}")


if __name__ == "__main__":
    main()
