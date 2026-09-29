"""Precompute projections at image build time so the Space starts instantly and demo mode
can serve UMAP (which it won't compute on the fly)."""

import sys
from pathlib import Path

from faissight.session import Session

data = Path(sys.argv[1])
for name in ("ivf_pq", "ivf_flat", "hnsw_flat"):
    session = Session(data / f"{name}.index", vectors=data / "vectors.npy")
    for method, dims in (("pca", 2), ("pca", 3), ("umap", 2)):
        job = session.projection_job(method, dims)
        job.wait()
        if job.error is not None:
            raise job.error
        print(f"{name}: {method} {dims}d cached", flush=True)
