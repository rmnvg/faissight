"""Saved sweep runs and baseline comparison: keep an experiment, then check for regressions.

A run record is plain JSON (see :data:`RUN_FORMAT`): the index identity (sha1 of its bytes),
the query set's fingerprint, the sweep settings, the environment the latency was measured
in, every measurement, the recommendation for a target, and the worst queries. Two records
can be compared setting by setting, and CI can fail when recall drops or p95 latency grows
beyond a threshold.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import dataclass, field
from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np

from faissight.core.sweep import SweepResult
from faissight.core.types import LoadedIndex
from faissight.core.vectors import VectorSource

RUN_FORMAT = "faissight.sweep-run"
RUN_VERSION = 1

# Environment keys that change single-threaded latency; recall doesn't depend on them.
_LATENCY_ENV_KEYS = ("faiss", "system", "machine", "processor", "threads")

# Rows sampled (evenly spaced) to fingerprint a raw vector corpus: enough to catch a
# different or reordered --vectors file without hashing what can be gigabytes of data.
_CORPUS_SAMPLE_ROWS = 4096


class RunFormatError(ValueError):
    """The file is not a faissight sweep run this version can read."""


def _faissight_version() -> str:
    try:
        return metadata.version("faissight")
    except metadata.PackageNotFoundError:
        return "0.0.0+unknown"


def corpus_fingerprint(source: VectorSource) -> str:
    """A cheap identity for a raw-vector ground-truth corpus.

    Hashes shape/dtype plus up to :data:`_CORPUS_SAMPLE_ROWS` evenly spaced rows, not the
    whole array: the corpus a sweep's ``--vectors`` points at can be gigabytes, and this
    only needs to catch "you compared against a different file", not verify it exactly.
    """
    h = hashlib.sha1()
    h.update(f"{source.vectors.shape}:{source.vectors.dtype}".encode())
    n = len(source)
    if n:
        rows = np.linspace(0, n - 1, min(n, _CORPUS_SAMPLE_ROWS)).astype(np.int64)
        h.update(np.ascontiguousarray(source.vectors[rows], dtype=np.float32).tobytes())
        h.update(np.ascontiguousarray(source.ids[rows], dtype=np.int64).tobytes())
    return h.hexdigest()


def run_record(
    result: SweepResult,
    li: LoadedIndex,
    index_sha1: str,
    *,
    target_recall: float,
    confident: bool = False,
    max_p95_ms: float | None = None,
    label: str | None = None,
    source: VectorSource | None = None,
) -> dict[str, Any]:
    """A JSON-ready record of a finished sweep (:func:`save_run` writes it).

    ``source`` is the raw vectors ground truth was computed on (omit when
    ``result.truth_reconstructed``, where the index itself is the ground-truth identity):
    it adds a corpus fingerprint so two runs against the same query set but a different
    ``--vectors`` file aren't mistaken for comparable.
    """
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
            "ground_truth_fingerprint": corpus_fingerprint(source) if source is not None else None,
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


def _reject_constant(token: str) -> float:
    # json.loads accepts NaN/Infinity/-Infinity by default (a non-standard extension);
    # a run record with one would silently break every recall/latency comparison below.
    raise RunFormatError(f"Not valid JSON: {token} is not allowed in a sweep run.")


def load_run(path: str | Path) -> dict[str, Any]:
    """Read and check a run record written by :func:`save_run` (or the Tuner's Save run)."""
    try:
        data = json.loads(Path(path).read_text(), parse_constant=_reject_constant)
    except json.JSONDecodeError as e:
        raise RunFormatError(f"{path} is not JSON: {e}") from None
    return check_run(data)


def _finite(x: Any) -> float:
    v = float(x)
    if not math.isfinite(v):
        raise ValueError("not a finite number")
    return v


def _fraction(x: Any) -> float:
    v = _finite(x)
    if not -1e-9 <= v <= 1 + 1e-9:
        raise ValueError("outside [0, 1]")
    return v


def _nonneg(x: Any) -> float:
    v = _finite(x)
    if v < 0:
        raise ValueError("negative")
    return v


def _nonempty_str(x: Any) -> str:
    if not isinstance(x, str) or not x:
        raise ValueError("must be a non-empty string")
    return x


def check_run(data: Any) -> dict[str, Any]:
    """Validate a run record: not just its shape, but that its numbers can be trusted.

    Every field :func:`compare_runs` reads for a regression check is required and range
    checked (finite, recall in ``[0, 1]``, latency non-negative), point values must be
    unique, and the fingerprints comparisons key off must be present. A file that fails
    this is never silently compared against or saved over.
    """
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
        _nonempty_str(data["index"]["sha1"])
        _nonempty_str(data["settings"]["param"])
        int(data["settings"]["k"])
        _nonempty_str(data["queries"]["sha256"])
        if not isinstance(data["points"], list) or not data["points"]:
            raise ValueError("points must be a non-empty list")
        seen_values: set[int] = set()
        for p in data["points"]:
            value = int(p["value"])
            if value in seen_values:
                raise ValueError(f"duplicate point value {value}")
            seen_values.add(value)
            _fraction(p["recall"])
            _nonneg(p["latency_p95_ms"])
            if p.get("latency_mean_ms") is not None:
                _nonneg(p["latency_mean_ms"])
            if p.get("probe_coverage") is not None:
                _fraction(p["probe_coverage"])
            lo, hi = p.get("recall_ci_low"), p.get("recall_ci_high")
            if lo is not None and hi is not None and _fraction(lo) > _fraction(hi) + 1e-9:
                raise ValueError("recall_ci_low > recall_ci_high")
    except (KeyError, TypeError, ValueError) as e:
        raise RunFormatError(f"Incomplete or invalid sweep run: {e}.") from None
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

    @property
    def comparable(self) -> bool:
        """False when nothing was actually checked: no shared, judgeable settings.

        This is distinct from ``regressed``: a comparison that couldn't check anything
        (different query sets, no overlapping values, incompatible metrics, ...) must not
        be mistaken for a clean pass. Callers gating CI should treat this as its own
        failure, separate from ``regressed``.
        """
        return bool(self.points)


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
    base_fp, cur_fp = bs.get("ground_truth_fingerprint"), cs.get("ground_truth_fingerprint")
    if base_fp is not None and cur_fp is not None and base_fp != cur_fp:
        notes.append(
            "Different ground-truth corpus (the --vectors file differs): recall isn't comparable."
        )
        recall_ok = False
    if base["index"].get("metric") != cur["index"].get("metric"):
        notes.append(
            f"Different metrics ({base['index'].get('metric')} vs {cur['index'].get('metric')}): "
            "recall isn't comparable."
        )
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
