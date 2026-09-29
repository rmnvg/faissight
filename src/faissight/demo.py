"""The RAG demo: real passages -> MiniLM embeddings -> IVFFlat, IVFPQ and HNSW indexes.

Used by ``faissight demo`` and ``examples/make_rag_demo.py``. Needs the ``text`` and
``parquet`` extras (sentence-transformers, pyarrow).

Data: "RAG Dataset 12000" by Neural Bridge AI
(https://huggingface.co/datasets/neural-bridge/rag-dataset-12000), Apache-2.0. Its
passages come from Falcon RefinedWeb (tiiuae/falcon-refinedweb, ODC-By 1.0). The parquet
files are downloaded directly (no ``datasets`` dependency) and cached next to the output.
"""

from __future__ import annotations

import json
import time
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np

DATASET = "neural-bridge/rag-dataset-12000"
BASE = f"https://huggingface.co/datasets/{DATASET}/resolve/main/data/"
FILES = (
    "train-00000-of-00001-9df3a936e1f63191.parquet",
    "test-00000-of-00001-af2a9f454ad1b8a3.parquet",
)
MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_WORDS = 150  # ~200 tokens for English prose
SOURCE = f"{DATASET} (Apache-2.0); text from Falcon RefinedWeb (ODC-By 1.0)"
DATA_LICENSE = f"""RAG demo data

Passages and questions: "RAG Dataset 12000" by Neural Bridge AI
  https://huggingface.co/datasets/{DATASET}
  License: Apache License 2.0 (https://www.apache.org/licenses/LICENSE-2.0)

The passages originate from Falcon RefinedWeb (tiiuae/falcon-refinedweb),
  released under the Open Data Commons Attribution License (ODC-By) v1.0.

faissight chunked the passages and embedded them with {MODEL} (Apache-2.0).
"""


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def download(out: Path) -> list[Path]:
    """Fetch the parquet files once; later runs reuse the cached copies."""
    raw = out / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    paths = []
    for name in FILES:
        path = raw / name
        if not path.exists():
            log(f"downloading {name}")
            tmp = path.with_suffix(".part")
            urllib.request.urlretrieve(BASE + name, tmp)
            tmp.rename(path)
        paths.append(path)
    return paths


def load_rows(paths: list[Path]) -> tuple[list[str], list[str]]:
    """Unique passages (in file order) and all questions."""
    import pyarrow.parquet as pq

    contexts: dict[str, None] = {}
    questions: list[str] = []
    for path in paths:
        table = pq.read_table(path, columns=["context", "question"]).to_pydict()
        for ctx, q in zip(table["context"], table["question"], strict=True):
            if ctx and ctx.strip():
                contexts.setdefault(" ".join(ctx.split()))
            if q and q.strip():
                questions.append(q.strip())
    return list(contexts), questions


def chunk(passages: list[str], max_chunks: int) -> list[dict[str, object]]:
    """Split passages into ~CHUNK_WORDS-word chunks; each chunk keeps its passage id."""
    chunks: list[dict[str, object]] = []
    for doc_id, text in enumerate(passages):
        words = text.split()
        for start in range(0, len(words), CHUNK_WORDS):
            piece = words[start : start + CHUNK_WORDS]
            if len(piece) < 20 and start > 0:
                continue  # drop tiny tails
            chunks.append(
                {
                    "id": len(chunks),
                    "text": " ".join(piece),
                    "title": " ".join(words[:10]) + ("…" if len(words) > 10 else ""),
                    "doc_id": doc_id,
                    "source": SOURCE,
                }
            )
            if len(chunks) >= max_chunks:
                return chunks
    return chunks


def embed(texts: list[str], batch_size: int = 256) -> np.ndarray:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(MODEL)
    vecs = model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True,
    )
    return np.ascontiguousarray(vecs, dtype=np.float32)


def build_indexes(x: np.ndarray, nlist: int, pq_m: int, hnsw_m: int) -> dict[str, Any]:
    from faissight.core._faiss import import_faiss

    faiss = import_faiss()
    d = x.shape[1]
    ip = faiss.METRIC_INNER_PRODUCT
    out: dict[str, Any] = {}

    ivf_flat = faiss.IndexIVFFlat(faiss.IndexFlatIP(d), d, nlist, ip)
    ivf_flat.train(x)
    ivf_flat.add(x)
    out["ivf_flat"] = ivf_flat

    ivf_pq = faiss.IndexIVFPQ(faiss.IndexFlatIP(d), d, nlist, pq_m, 8, ip)
    ivf_pq.train(x)
    ivf_pq.add(x)
    out["ivf_pq"] = ivf_pq

    hnsw = faiss.IndexHNSWFlat(d, hnsw_m, ip)
    hnsw.add(x)
    out["hnsw_flat"] = hnsw
    return out


def default_dir() -> Path:
    """Where ``faissight demo`` keeps its data (next to the projection cache)."""
    from faissight.core.projection import default_cache_root

    return default_cache_root() / "demo-rag"


def is_built(out: Path) -> bool:
    return all(
        (out / f).exists() for f in (*INDEX_FILES, "vectors.npy", "chunks.jsonl", "queries.npy")
    )


INDEX_FILES = ("ivf_flat.index", "ivf_pq.index", "hnsw_flat.index")


def build_demo(
    out: Path,
    *,
    max_chunks: int = 10_000,
    n_queries: int = 200,
    nlist: int = 256,
    pq_m: int = 48,
    hnsw_m: int = 32,
    seed: int = 0,
) -> Path:
    """Download, chunk, embed and index the demo corpus into ``out``. Returns ``out``."""
    from faissight.core._faiss import import_faiss

    faiss = import_faiss()
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()

    passages, questions = load_rows(download(out))
    chunks = chunk(passages, max_chunks)
    log(f"{len(passages):,} passages -> {len(chunks):,} chunks; {len(questions):,} questions")

    log(f"embedding chunks with {MODEL}")
    x = embed([str(c["text"]) for c in chunks])
    rng = np.random.default_rng(seed)
    n_q = min(n_queries, len(questions))
    q_idx = np.sort(rng.choice(len(questions), size=n_q, replace=False))
    q_text = [questions[i] for i in q_idx]
    log(f"embedding {len(q_text)} questions")
    q = embed(q_text)

    log("building indexes")
    # Fewer lists for small corpora keeps ~40+ vectors per list for IVF training.
    nlist = min(nlist, max(16, len(chunks) // 40))
    for name, index in build_indexes(x, nlist, pq_m, hnsw_m).items():
        faiss.write_index(index, str(out / f"{name}.index"))

    np.save(out / "vectors.npy", x)
    np.save(out / "queries.npy", q)
    with (out / "chunks.jsonl").open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    with (out / "queries.jsonl").open("w", encoding="utf-8") as f:
        for i, text in enumerate(q_text):
            f.write(json.dumps({"id": i, "text": text}, ensure_ascii=False) + "\n")
    (out / "DATA_LICENSE.txt").write_text(DATA_LICENSE)
    log(f"done in {time.perf_counter() - t0:.0f}s -> {out}")
    return out
