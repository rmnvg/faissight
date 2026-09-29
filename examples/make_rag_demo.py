"""Build the RAG demo data into a folder you choose (``faissight demo`` does this for you).

    uv run python examples/make_rag_demo.py --out examples/rag_data --max-chunks 20000
    uv run faissight serve examples/rag_data/ivf_pq.index \\
        --vectors examples/rag_data/vectors.npy --meta examples/rag_data/chunks.jsonl \\
        --queries examples/rag_data/queries.npy --embedder all-MiniLM-L6-v2

See ``faissight.demo`` for the data source and licence.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from faissight.demo import build_demo


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
    build_demo(
        args.out,
        max_chunks=args.max_chunks,
        n_queries=args.n_queries,
        nlist=args.nlist,
        pq_m=args.pq_m,
        hnsw_m=args.hnsw_m,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
