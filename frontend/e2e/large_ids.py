"""Small real HNSW fixture with adjacent ids beyond JavaScript's exact integer range."""

import json
from pathlib import Path

import faiss
import numpy as np

out = Path("e2e/.data-large")
out.mkdir(parents=True, exist_ok=True)
x = np.random.default_rng(6).normal(size=(100, 8)).astype(np.float32)
ids = np.arange(len(x), dtype=np.int64) + 2**53 + 1
index = faiss.IndexIDMap2(faiss.IndexHNSWFlat(8, 12))
index.add_with_ids(x, ids)
faiss.write_index(index, str(out / "hnsw.index"))
np.save(out / "vectors.npy", x)
np.save(out / "ids.npy", ids)
(out / "metadata.jsonl").write_text(
    "\n".join(json.dumps({"id": int(i), "text": f"Passage {i}"}) for i in ids)
)
