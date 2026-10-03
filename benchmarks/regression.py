"""Performance regression checks: the plan's budgets, plus a same-machine A/B comparison.

    uv run python benchmarks/regression.py measure --output current.json
    uv run python benchmarks/regression.py check current.json [--baseline baseline.json]

``measure`` uses only the public ``Session`` API, so this file can measure an older
checkout too: run it with that checkout's interpreter (see .github/workflows/performance.yml).
Absolute budgets are the plan's (20k x 384 on a laptop CPU, PLAN.md section 10) and are
generous on purpose. Real regressions are caught by comparing with a baseline measured on
the same runner, which cancels out how fast that runner happens to be.
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
import platform
import resource
import statistics
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

N, D, NLIST, SEED = 20_000, 384, 128, 42
SWEEP_VALUES = [1, 2, 4, 8, 16, 32, 64, 128]


@dataclass(frozen=True)
class Budget:
    """An absolute ceiling, and how much worse than the baseline is still tolerated."""

    limit: float
    max_ratio: float
    """Fail when current > baseline x max_ratio ..."""
    min_delta: float
    """... and the difference is at least this much (ignores noise on tiny values)."""


BUDGETS: dict[str, Budget] = {
    "startup_s": Budget(3.0, 2.0, 0.25),
    "pca_s": Budget(2.0, 2.0, 0.25),
    "first_query_ms": Budget(1000.0, 2.0, 50.0),
    "query_p95_ms": Budget(150.0, 2.0, 5.0),
    "sweep_s": Budget(20.0, 2.0, 0.5),
    # The documented RAM formula (index + n x d x 4 + ~100 MB) is ~160 MiB here.
    "peak_rss_mib": Budget(512.0, 1.25, 32.0),
}


def _timed(fn: Callable[[], Any]) -> tuple[Any, float]:
    start = time.perf_counter()
    out = fn()
    return out, time.perf_counter() - start


def _supported(fn: Callable[..., Any], **kwargs: Any) -> dict[str, Any]:
    """The keyword arguments ``fn`` accepts: older baselines lack newer options."""
    params = inspect.signature(fn).parameters
    return {k: v for k, v in kwargs.items() if k in params}


def _close(session: Any) -> None:
    close = getattr(session.jobs, "close", None)  # added after the first prototype (dcc9060)
    if close is not None:
        close()


def _peak_rss_mib() -> float:
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss / (1024**2 if platform.system() == "Darwin" else 1024)


def write_dataset(out: Path) -> None:
    """Write the fixed corpus, queries and IVF index as files, like a user's inputs."""
    import faiss
    import numpy as np

    rng = np.random.default_rng(SEED)
    # Clustered data, like real embeddings, so IVF lists and recall curves are realistic.
    centers = rng.normal(size=(64, D)).astype("float32")

    def sample(n: int) -> Any:
        return centers[rng.integers(0, 64, n)] + 0.3 * rng.normal(size=(n, D)).astype("float32")

    vectors, queries = sample(N), sample(200)
    index = faiss.IndexIVFFlat(faiss.IndexFlatL2(D), D, NLIST)
    index.train(vectors)
    index.add(vectors)
    faiss.write_index(index, str(out / "index.ivf"))
    np.save(out / "vectors.npy", vectors)
    np.save(out / "queries.npy", queries)


def measure(repeats: int = 3) -> dict[str, Any]:
    """Measure one fresh process: startup, PCA, cold and warm queries, a sweep, peak RSS.

    The dataset is written by a child process, so generating it doesn't count towards
    this process's peak RSS. Timings that can be repeated without caches skewing them are
    medians of ``repeats``.
    """
    import faiss
    import numpy as np

    import faissight
    from faissight.core.projection import ProjectionMethod
    from faissight.session import Session

    faiss.omp_set_num_threads(1)
    with tempfile.TemporaryDirectory() as tmp:
        data = Path(tmp)
        subprocess.run([sys.executable, __file__, "dataset", str(data)], check=True)
        index, vectors, queries = data / "index.ivf", data / "vectors.npy", data / "queries.npy"
        query_vectors = np.load(queries)

        def open_session() -> Session:
            return Session(
                index, vectors=vectors, queries=queries, cache_root=data, disk_cache=False
            )

        startups = []
        for _ in range(repeats):
            session, seconds = _timed(open_session)
            startups.append(seconds)
            _close(session)
        session = open_session()
        try:
            job, pca_s = _timed(lambda: session.projection_job(ProjectionMethod.PCA, 2))
            _, waited = _timed(job.wait)
            if job.error is not None:
                raise job.error
            _, first_query_s = _timed(lambda: session.query(vector=query_vectors[0], nprobe=8))
            warm = [
                _timed(lambda q=q: session.query(vector=q, nprobe=8))[1] * 1000
                for q in query_vectors[1:51]
            ]
        finally:
            _close(session)
        # Each sweep gets a fresh session: a repeated sweep would be served from its cache,
        # and only newer versions take a seed. One timed pass after a warm-up, as in the first
        # prototype (dcc9060).
        sweeps = []
        for _ in range(repeats):
            session = open_session()
            try:
                _, job = session.sweep_job(
                    **_supported(session.sweep_job, values=SWEEP_VALUES, n_queries=200, repeats=1)
                )
                _, seconds = _timed(lambda job=job: job.wait(120))
                if job.error is not None:
                    raise job.error
                sweeps.append(seconds)
            finally:
                _close(session)

    return {
        "faissight": getattr(faissight, "__version__", "unknown"),
        "faiss": faiss.__version__,
        "numpy": np.__version__,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "dataset": {"n": N, "d": D, "nlist": NLIST, "seed": SEED, "threads": 1},
        "metrics": {
            "startup_s": statistics.median(startups),
            "pca_s": pca_s + waited,
            "first_query_ms": first_query_s * 1000,
            "query_p95_ms": float(np.percentile(warm, 95)),
            "sweep_s": statistics.median(sweeps),
            "peak_rss_mib": _peak_rss_mib(),
        },
    }


def check(current: dict[str, float], baseline: dict[str, float] | None = None) -> list[str]:
    """Budget violations: absolute ceilings, and slowdowns against ``baseline`` if given."""
    failures = []
    for key, budget in BUDGETS.items():
        value = current.get(key)
        if value is None:
            continue
        if value > budget.limit:
            failures.append(f"{key} = {value:.3g} exceeds the budget of {budget.limit:g}")
        base = (baseline or {}).get(key)
        if (
            base is not None
            and value > base * budget.max_ratio
            and value - base >= budget.min_delta
        ):
            failures.append(
                f"{key} = {value:.3g} is {value / base:.1f}x the baseline ({base:.3g}); "
                f"allowed {budget.max_ratio:g}x"
            )
    return failures


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    m = sub.add_parser("measure", help="Measure this interpreter's faissight.")
    m.add_argument("--output", type=Path, required=True)
    m.add_argument("--repeats", type=int, default=3)
    d = sub.add_parser("dataset", help="Write the benchmark inputs (used by measure).")
    d.add_argument("out", type=Path)
    c = sub.add_parser("check", help="Check measurements against budgets and a baseline.")
    c.add_argument("current", type=Path)
    c.add_argument("--baseline", type=Path)
    args = parser.parse_args(argv)

    if args.command == "dataset":
        write_dataset(args.out)
        return 0
    if args.command == "measure":
        result = measure(args.repeats)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result["metrics"], indent=2))
        return 0

    current = json.loads(args.current.read_text())
    baseline = json.loads(args.baseline.read_text()) if args.baseline else None
    rows = []
    for key in BUDGETS:
        now = current["metrics"].get(key)
        then = baseline["metrics"].get(key) if baseline else None
        rows.append(f"{key:16} {now!s:>24} {'' if then is None else then!s:>24}")
    print(f"{'metric':16} {'current':>24} {'baseline' if baseline else '':>24}")
    print("\n".join(rows))
    failures = check(current["metrics"], baseline["metrics"] if baseline else None)
    for failure in failures:
        # GitHub Actions turns ::error:: lines into annotations on the run.
        print(f"::error::{failure}" if os.environ.get("GITHUB_ACTIONS") else f"FAIL {failure}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
