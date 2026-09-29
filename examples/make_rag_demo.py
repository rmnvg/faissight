"""Build the RAG demo: real passages -> embeddings -> IVFFlat, IVFPQ and HNSW indexes.

    uv run python examples/make_rag_demo.py            # needs faissight[text,parquet]
    uv run faissight serve examples/rag_data/ivf_pq.index \\
        --vectors examples/rag_data/vectors.npy --meta examples/rag_data/chunks.jsonl \\
        --queries examples/rag_data/queries.npy --embedder all-MiniLM-L6-v2

Data: "RAG Dataset 12000" by Neural Bridge AI
(https://huggingface.co/datasets/neural-bridge/rag-dataset-12000), Apache-2.0. Its
passages come from Falcon RefinedWeb (tiiuae/falcon-refinedweb, ODC-By 1.0). The parquet
files are downloaded directly (no ``datasets`` dependency) and cached under ``--out``.

Writes into ``--out``:

- ``ivf_flat.index`` (IVF256,Flat), ``ivf_pq.index`` (IVF256,PQ48x8), ``hnsw_flat.index``
  (HNSW32), all inner product on L2-normalised embeddings
- ``vectors.npy`` (float32, n x 384), ``chunks.jsonl`` (id, text, title, doc_id, source)
- ``queries.npy`` + ``queries.jsonl``: embedded questions from the dataset
- ``DATA_LICENSE.txt``: attribution for the text
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.request
from pathlib import Path

import faiss
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


def build_indexes(x: np.ndarray, nlist: int, pq_m: int, hnsw_m: int) -> dict[str, faiss.Index]:
    d = x.shape[1]
    ip = faiss.METRIC_INNER_PRODUCT
    out: dict[str, faiss.Index] = {}

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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=Path(__file__).parent / "rag_data")
    parser.add_argument("--max-chunks", type=int, default=20_000)
    parser.add_argument("--n-queries", type=int, default=200)
    parser.add_argument("--nlist", type=int, default=256)
    parser.add_argument("--pq-m", type=int, default=48)
    parser.add_argument("--hnsw-m", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()

    passages, questions = load_rows(download(out))
    chunks = chunk(passages, args.max_chunks)
    log(f"{len(passages):,} passages -> {len(chunks):,} chunks; {len(questions):,} questions")

    log(f"embedding chunks with {MODEL}")
    x = embed([str(c["text"]) for c in chunks])
    rng = np.random.default_rng(args.seed)
    n_q = min(args.n_queries, len(questions))
    q_idx = np.sort(rng.choice(len(questions), size=n_q, replace=False))
    q_text = [questions[i] for i in q_idx]
    log(f"embedding {len(q_text)} questions")
    q = embed(q_text)

    log("building indexes")
    for name, index in build_indexes(x, args.nlist, args.pq_m, args.hnsw_m).items():
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


if __name__ == "__main__":
    main()
