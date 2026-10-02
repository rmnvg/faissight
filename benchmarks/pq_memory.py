"""Compare reconstruction-error workspace in isolated processes.

Run: python benchmarks/pq_memory.py [batched|previous]
Peak RSS includes Python and both input arrays; it is not total application memory.
"""

import resource
import sys

import numpy as np

from faissight.core.pq import reconstruction_errors
from faissight.core.vectors import VectorSource

rng = np.random.default_rng(0)
x = rng.standard_normal((100_000, 128), dtype=np.float32)
stored = x + np.float32(0.1)
ids = np.arange(len(x), dtype=np.int64)
mode = sys.argv[1] if len(sys.argv) > 1 else "batched"
if mode == "previous":
    raw64 = x.astype(np.float64)
    diff = raw64 - stored.astype(np.float64)
    errors = np.einsum("ij,ij->i", diff, diff)
    norms = np.einsum("ij,ij->i", raw64, raw64)
    relative = np.divide(errors, norms, out=np.zeros_like(errors), where=norms > 0)
elif mode == "batched":
    errors, relative = reconstruction_errors(
        VectorSource(x, ids, False), VectorSource(stored, ids, True)
    )
else:
    raise SystemExit("Choose batched or previous.")
peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
mib = peak / (1024**2 if sys.platform == "darwin" else 1024)
print(f"{mode}: peak RSS {mib:.1f} MiB; mean squared error {errors.mean():.8f}")
