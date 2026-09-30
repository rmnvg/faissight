"""Peak memory of HNSW inspection on a large synthetic index.

Each scenario runs in a fresh process: load the index (and memory-mapped vectors), then
measure the peak extra memory of one operation. On macOS this is the physical footprint's
interval peak (``proc_pid_rusage``, reset right before the operation); elsewhere it falls
back to the lifetime peak RSS, which also counts loading.

    uv run python benchmarks/hnsw_memory.py --out /tmp/hnsw-bench --n 1000000 --d 64
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

SCENARIOS = ("stats", "graph", "trace")


class _RusageInfoV4(ctypes.Structure):
    # <sys/resource.h> struct rusage_info_v4, fields up to ri_interval_max_phys_footprint.
    _fields_ = [
        ("ri_uuid", ctypes.c_uint8 * 16),
        *[
            (name, ctypes.c_uint64)
            for name in [
                "ri_user_time",
                "ri_system_time",
                "ri_pkg_idle_wkups",
                "ri_interrupt_wkups",
                "ri_pageins",
                "ri_wired_size",
                "ri_resident_size",
                "ri_phys_footprint",
                "ri_proc_start_abstime",
                "ri_proc_exit_abstime",
                "ri_child_user_time",
                "ri_child_system_time",
                "ri_child_pkg_idle_wkups",
                "ri_child_interrupt_wkups",
                "ri_child_pageins",
                "ri_child_elapsed_abstime",
                "ri_diskio_bytesread",
                "ri_diskio_byteswritten",
                "ri_cpu_time_qos_default",
                "ri_cpu_time_qos_maintenance",
                "ri_cpu_time_qos_background",
                "ri_cpu_time_qos_utility",
                "ri_cpu_time_qos_legacy",
                "ri_cpu_time_qos_user_initiated",
                "ri_cpu_time_qos_user_interactive",
                "ri_billed_system_time",
                "ri_serviced_system_time",
                "ri_logical_writes",
                "ri_lifetime_max_phys_footprint",
                "ri_instructions",
                "ri_cycles",
                "ri_billed_energy",
                "ri_serviced_energy",
                "ri_interval_max_phys_footprint",
            ]
        ],
        ("ri_runnable_time", ctypes.c_uint64),
    ]


def _footprint() -> tuple[int, int] | None:
    """(current footprint, peak since the previous call) in bytes; None off macOS."""
    if sys.platform != "darwin":
        return None
    info = _RusageInfoV4()
    libc = ctypes.CDLL("/usr/lib/libSystem.B.dylib")
    if libc.proc_pid_rusage(os.getpid(), 4, ctypes.byref(info)) != 0:
        return None
    return info.ri_phys_footprint, info.ri_interval_max_phys_footprint


def build(out: Path, n: int, d: int, m: int) -> None:
    import faiss

    out.mkdir(parents=True, exist_ok=True)
    if (out / "hnsw.index").exists():
        return
    rng = np.random.default_rng(0)
    x = rng.standard_normal((n, d), dtype=np.float32)
    np.save(out / "vectors.npy", x)
    t0 = time.perf_counter()
    index = faiss.IndexHNSWFlat(d, m)
    index.add(x)
    faiss.write_index(index, str(out / "hnsw.index"))
    print(f"built HNSW{m} over {n} x {d} in {time.perf_counter() - t0:.0f} s", file=sys.stderr)


def run(out: Path, scenario: str) -> dict[str, float | str]:
    from faissight.core import hnsw as H
    from faissight.session import Session

    s = Session(out / "hnsw.index", vectors=out / "vectors.npy", mmap=True, disk_cache=False)
    _ = s.li
    if scenario != "stats":
        # The layout uses the session's PCA projection; fit it before measuring.
        job = s.projection_job()
        job.wait()
    before = _footprint()
    t0 = time.perf_counter()
    if scenario == "stats":
        H.graph_stats(s.hnsw_graph)
    elif scenario == "graph":
        g = s.hnsw_graph
        nodes = H.neighbourhood(g, [g.entry_point], 0, 2000)
        g.edges(0, nodes)
        s.hnsw_positions(nodes)
    else:
        q = np.load(out / "vectors.npy", mmap_mode="r")[:1] + 0.01
        s.hnsw_trace(q[0], 10, 64)
    elapsed = time.perf_counter() - t0
    after = _footprint()
    result: dict[str, float | str] = {"scenario": scenario, "seconds": round(elapsed, 3)}
    if before is not None and after is not None:
        result["peak_extra_mb"] = round((after[1] - before[0]) / 2**20, 1)
    else:
        import resource

        result["peak_rss_mb"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20, 1)
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n", type=int, default=1_000_000)
    ap.add_argument("--d", type=int, default=64)
    ap.add_argument("--m", type=int, default=16)
    ap.add_argument("--scenario", choices=SCENARIOS, help="Run one scenario in this process.")
    args = ap.parse_args()
    if args.scenario:
        print(json.dumps(run(args.out, args.scenario)))
        return
    build(args.out, args.n, args.d, args.m)
    for scenario in SCENARIOS:
        cmd = [sys.executable, __file__, "--out", str(args.out), "--scenario", scenario]
        print(subprocess.run(cmd, check=True, capture_output=True, text=True).stdout.strip())


if __name__ == "__main__":
    main()
