"""Saved sweep runs and baseline comparison: keep an experiment, then check for regressions.

A run record is plain JSON (see :data:`RUN_FORMAT`): the index identity (sha1 of its bytes),
the query set's fingerprint, the sweep settings, the environment the latency was measured
in, every measurement, the recommendation for a target, and the worst queries. Two records
can be compared setting by setting, and CI can fail when recall drops or p95 latency grows
beyond a threshold.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from importlib import metadata
from pathlib import Path
from typing import Any

from faissight.core.sweep import SweepResult
from faissight.core.types import LoadedIndex

RUN_FORMAT = "faissight.sweep-run"
RUN_VERSION = 1

# Environment keys that change single-threaded latency; recall doesn't depend on them.
_LATENCY_ENV_KEYS = ("faiss", "system", "machine", "processor", "threads")


class RunFormatError(ValueError):
    """The file is not a faissight sweep run this version can read."""


def _faissight_version() -> str:
    try:
        return metadata.version("faissight")
    except metadata.PackageNotFoundError:
        return "0.0.0+unknown"


def run_record(
    result: SweepResult,
    li: LoadedIndex,
    index_sha1: str,
    *,
    target_recall: float,
    confident: bool = False,
    max_p95_ms: float | None = None,
    label: str | None = None,
) -> dict[str, Any]:
    """A JSON-ready record of a finished sweep (:func:`save_run` writes it)."""
    choice = result.choose(target_recall, confident=confident, max_p95_ms=max_p95_ms)
    return {
        "format": RUN_FORMAT,
        "version": RUN_VERSION,
        "label": label,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "faissight": _faissight_version(),
        "index": {
            "sha1": index_sha1,
            "name": li.path.name if li.path is not None else None,
            "kind": li.kind.value,
            "class_chain": list(li.class_chain),
            "ntotal": li.ntotal,
            "d": li.d,
            "metric": li.metric.value,
            "params": li.params.as_dict(),
        },
        "queries": {
            "origin": result.query_origin,
            "n": result.n_queries,
            "sha256": result.query_sha256,
            "seed": result.query_seed,
        },
        "settings": {
            "param": result.param.value,
            "values": [p.value for p in result.points],
            "k": result.k,
            "repeats": result.repeats,
            "seed": result.seed,
            "truth_source": "reconstructed" if result.truth_reconstructed else "raw",
        },
        "environment": dict(result.environment),
        "decision": {
            "target_recall": target_recall,
            "confident": confident,
            "max_p95_ms": max_p95_ms,
            "status": choice.status,
            "recommended": choice.point.value if choice.point else None,
        },
        "points": [
            {
                "value": p.value,
                "recall": p.recall,
                "recall_ci_low": p.recall_ci_low,
                "recall_ci_high": p.recall_ci_high,
                "latency_mean_ms": p.latency_mean_ms,
                "latency_p95_ms": p.latency_p95_ms,
                "probe_coverage": p.probe_coverage,
                "recall_distribution": [list(pair) for pair in p.recall_distribution],
                "worst_queries": [
                    {"query_no": w.query_no, "id": w.id, "recall": w.recall}
                    for w in p.worst_queries
                ],
            }
            for p in result.points
        ],
    }


def save_run(record: dict[str, Any], path: str | Path) -> Path:
    """Write a run record as indented JSON."""
    out = Path(path)
    out.write_text(json.dumps(check_run(record), indent=2) + "\n")
    return out


def load_run(path: str | Path) -> dict[str, Any]:
    """Read and check a run record written by :func:`save_run` (or the Tuner's Save run)."""
    try:
        data = json.loads(Path(path).read_text())
    except json.JSONDecodeError as e:
        raise RunFormatError(f"{path} is not JSON: {e}") from None
    return check_run(data)


def check_run(data: Any) -> dict[str, Any]:
    """Validate the parts of a run record that comparisons rely on; return it."""
    if not isinstance(data, dict) or data.get("format") != RUN_FORMAT:
        raise RunFormatError("Not a faissight sweep run (expected format faissight.sweep-run).")
    if data.get("version") != RUN_VERSION:
        raise RunFormatError(
            f"Run format version {data.get('version')} isn't supported (this faissight reads "
            f"version {RUN_VERSION})."
        )
    try:
        for key in ("index", "queries", "settings", "environment", "decision"):
            if not isinstance(data[key], dict):
                raise TypeError(key)
        for p in data["points"]:
            int(p["value"])
            float(p["recall"])
            float(p["latency_p95_ms"])
        for section, key in (("settings", "param"), ("settings", "k"), ("index", "sha1")):
            if key not in data[section]:
                raise KeyError(f"{section}.{key}")
    except (KeyError, TypeError, ValueError) as e:
        raise RunFormatError(f"Incomplete sweep run: missing or invalid {e}.") from None
    return dict(data)


@dataclass(frozen=True)
class PointDelta:
    """One setting measured in both runs."""

    value: int
    baseline_recall: float
    recall: float
    baseline_p95_ms: float
    p95_ms: float
    recall_regressed: bool
    latency_regressed: bool

    @property
    def recall_change(self) -> float:
        return self.recall - self.baseline_recall

    @property
    def p95_change(self) -> float:
        """Relative change of p95 latency (0.2 = 20% slower)."""
        return self.p95_ms / self.baseline_p95_ms - 1 if self.baseline_p95_ms > 0 else 0.0


@dataclass(frozen=True)
class RunComparison:
    """``current`` against ``baseline``, setting by setting."""

    points: list[PointDelta]
    max_recall_drop: float
    max_p95_increase: float
    min_p95_increase_ms: float = 0.0
    notes: list[str] = field(default_factory=list)
    """Why the two runs may not be comparable (different index, queries, machine, ...)."""
    recall_comparable: bool = True
    """Same k, parameter and query set: recall differences mean the index changed."""
    latency_comparable: bool = True
    """Also measured in the same environment, so latency differences are meaningful."""
    baseline_recommended: int | None = None
    recommended: int | None = None

    @property
    def regressions(self) -> list[PointDelta]:
        return [p for p in self.points if p.recall_regressed or p.latency_regressed]

    @property
    def regressed(self) -> bool:
        return bool(self.regressions)


def compare_runs(
    baseline: dict[str, Any],
    current: dict[str, Any],
    *,
    max_recall_drop: float = 0.01,
    max_p95_increase: float = 0.2,
    min_p95_increase_ms: float = 0.05,
) -> RunComparison:
    """Compare two run records setting by setting.

    A setting regresses when its recall falls by more than ``max_recall_drop`` (absolute)
    or its p95 latency grows by more than ``max_p95_increase`` (relative) *and* by more than
    ``min_p95_increase_ms``: sub-millisecond timings vary by tens of percent between runs,
    and the floor keeps that noise from failing CI. Latency is only judged when both runs
    were measured in the same environment; recall only when they use the same parameter, k
    and query set. ``notes`` say what differs.
    """
    if max_recall_drop < 0 or max_p95_increase < 0 or min_p95_increase_ms < 0:
        raise ValueError("Regression thresholds must be non-negative.")
    base, cur = check_run(baseline), check_run(current)
    notes: list[str] = []
    bs, cs = base["settings"], cur["settings"]
    recall_ok = True
    if bs["param"] != cs["param"] or bs["k"] != cs["k"]:
        notes.append(
            f"Different sweeps: {bs['param']} at k={bs['k']} vs {cs['param']} at k={cs['k']}."
        )
        recall_ok = False
    if base["queries"].get("sha256") != cur["queries"].get("sha256"):
        notes.append("Different query sets (fingerprints differ): recall isn't comparable.")
        recall_ok = False
    if bs.get("truth_source") != cs.get("truth_source"):
        notes.append("One run's ground truth used decoded vectors, the other raw vectors.")
        recall_ok = False
    if base["index"]["sha1"] != cur["index"]["sha1"]:
        notes.append("Different index files (sha1 differs): changes reflect the new index.")
    env_diff = [
        k for k in _LATENCY_ENV_KEYS if base["environment"].get(k) != cur["environment"].get(k)
    ]
    latency_ok = not env_diff
    if env_diff:
        notes.append(
            f"Measured in different environments ({', '.join(env_diff)}): latency isn't "
            "comparable, so it isn't checked."
        )
    bd, cd = base["decision"], cur["decision"]
    if (bd.get("target_recall"), bd.get("max_p95_ms"), bd.get("confident")) != (
        cd.get("target_recall"),
        cd.get("max_p95_ms"),
        cd.get("confident"),
    ):
        notes.append("The runs were saved for different targets, so their recommendations differ.")
    base_points = {int(p["value"]): p for p in base["points"]}
    deltas = []
    for p in cur["points"]:
        b = base_points.get(int(p["value"]))
        if b is None or not recall_ok:
            continue
        recall_drop = float(b["recall"]) - float(p["recall"])
        b95, c95 = float(b["latency_p95_ms"]), float(p["latency_p95_ms"])
        deltas.append(
            PointDelta(
                value=int(p["value"]),
                baseline_recall=float(b["recall"]),
                recall=float(p["recall"]),
                baseline_p95_ms=b95,
                p95_ms=c95,
                recall_regressed=recall_drop > max_recall_drop + 1e-12,
                latency_regressed=latency_ok
                and b95 > 0
                and c95 / b95 - 1 > max_p95_increase
                and c95 - b95 > min_p95_increase_ms,
            )
        )
    if recall_ok and not deltas:
        notes.append("No setting was measured in both runs.")
    return RunComparison(
        points=sorted(deltas, key=lambda d: d.value),
        max_recall_drop=max_recall_drop,
        max_p95_increase=max_p95_increase,
        min_p95_increase_ms=min_p95_increase_ms,
        notes=notes,
        recall_comparable=recall_ok,
        latency_comparable=latency_ok,
        baseline_recommended=base["decision"].get("recommended"),
        recommended=cur["decision"].get("recommended"),
    )
