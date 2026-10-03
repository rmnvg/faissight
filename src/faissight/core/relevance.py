"""Labelled retrieval evaluation, distinct from agreement with exact vector search."""

from __future__ import annotations

import json
import math
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt

from faissight.core.search import rerank, search
from faissight.core.types import LoadedIndex
from faissight.core.vectors import VectorSource, validate_ids


@dataclass(frozen=True)
class RelevanceMetrics:
    recall: float
    mrr: float
    ndcg: float


def relevance_metrics(found: list[int], labels: dict[int, float], k: int) -> RelevanceMetrics:
    """Recall/MRR/nDCG@k; positive grades are relevant, nDCG uses linear gains.

    Unjudged results count as non-relevant. Duplicate hits receive no extra credit.
    Every query must have at least one positive judgement.
    """
    if k < 1:
        raise ValueError("k must be >= 1.")
    if any(not math.isfinite(g) or g < 0 for g in labels.values()):
        raise ValueError("Relevance grades must be finite and non-negative.")
    relevant = {i for i, g in labels.items() if g > 0}
    if not relevant:
        raise ValueError("Each query needs at least one positively labelled chunk.")
    scale = max(labels.values())
    gains: list[float] = []
    seen: set[int] = set()
    first = 0.0
    hits = 0
    for rank, i in enumerate(found[:k], 1):
        hit = i not in seen and i in relevant
        gain = labels.get(i, 0.0) / scale if i not in seen else 0.0
        seen.add(i)
        gains.append(gain / math.log2(rank + 1))
        if hit:
            hits += 1
            if first == 0:
                first = 1 / rank
    ideal = sum(
        (g / scale) / math.log2(rank + 1)
        for rank, g in enumerate(sorted(labels.values(), reverse=True)[:k], 1)
    )
    return RelevanceMetrics(hits / len(relevant), first, sum(gains) / ideal)


def load_judgements(path: str | Path, n_queries: int) -> list[dict[int, float]]:
    """JSONL: {"row": 0, "relevant": {"42": 2, "77": 1}}; one row per query."""
    return parse_judgements(Path(path).read_text(), n_queries)


def parse_judgements(text: str, n_queries: int) -> list[dict[int, float]]:
    """Validate JSONL judgements supplied by a file or a web client."""
    rows: dict[int, dict[int, float]] = {}
    for line_no, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
            row, raw = obj["row"], obj["relevant"]
            if type(row) is not int or not 0 <= row < n_queries or row in rows:
                raise ValueError("row must be unique and within the query set")
            if not isinstance(raw, dict):
                raise ValueError("relevant must map decimal chunk ids to grades")
            labels = {}
            for key, grade in raw.items():
                if not key.isascii() or not key.isdecimal() or not 0 <= int(key) < 2**63:
                    raise ValueError("chunk ids must be non-negative int64 decimals")
                if int(key) in labels:
                    raise ValueError("duplicate chunk id")
                if type(grade) not in (int, float):
                    raise ValueError("grades must be numbers")
                labels[int(key)] = float(grade)
            relevance_metrics([], labels, 1)
            rows[row] = labels
        except (ValueError, TypeError, KeyError, OverflowError) as e:
            raise ValueError(f"Invalid judgements on line {line_no}: {e}") from e
    if len(rows) != n_queries or n_queries == 0:
        raise ValueError("Supply exactly one judgement row for every evaluation query.")
    return [rows[i] for i in range(n_queries)]


def evaluate_relevance(
    li: LoadedIndex,
    queries: npt.ArrayLike,
    judgements: list[dict[int, float]],
    source: VectorSource,
    *,
    k: int = 10,
    nprobe: int | None = None,
    ef_search: int | None = None,
    candidates: int | None = None,
    progress: Callable[[float, str], None] | None = None,
) -> dict[str, object]:
    """Macro-average labelled metrics, with optional exact candidate reranking.

    Each query is searched once per mode; latency is exploratory, not a benchmark.
    Supplied queries are held-out vectors, so no stored id is implicitly excluded.
    """
    q = np.asarray(queries, dtype=np.float32)
    if q.ndim != 2 or q.shape[1] != li.d or not np.isfinite(q).all():
        raise ValueError(f"Queries must be finite and have shape (n, {li.d}).")
    if len(q) == 0 or len(q) != len(judgements):
        raise ValueError("Supply one judgement row per query, with at least one query.")
    if k < 1 or (candidates is not None and candidates < k):
        raise ValueError("Require k >= 1 and candidates >= k.")
    if source.reconstructed:
        raise ValueError("Evaluation needs raw vectors.")
    if source.vectors.shape != (li.ntotal, li.d):
        raise ValueError("Raw vectors must match the index's count and dimension.")
    validate_ids(li, source.ids)
    for labels in judgements:
        relevance_metrics([], labels, k)
        if any(type(i) is not int or not 0 <= i < 2**63 for i in labels):
            raise ValueError("Labelled ids must be non-negative int64 integers.")
        if (source.positions(list(labels)) < 0).any():
            raise ValueError("Labelled chunk ids must exist in the vector corpus.")
    rows = []
    base_metrics, reranked_metrics = [], []
    for row, (query, labels) in enumerate(zip(q, judgements, strict=True)):
        if progress is not None:
            progress(row / len(q), f"Evaluating query {row + 1} of {len(q)}")
        base = search(li, query, k, nprobe=nprobe, ef_search=ef_search)
        metrics = relevance_metrics(base.valid_ids.tolist(), labels, k)
        base_metrics.append(metrics)
        entry: dict[str, object] = {
            "row": row,
            "metrics": asdict(metrics),
            "latency_ms": base.latency_ms,
            "ids": [str(i) for i in base.valid_ids],
        }
        if candidates is not None:
            pool = search(li, query, candidates, nprobe=nprobe, ef_search=ef_search)
            ranked = rerank(source, query, pool, k)
            rm = relevance_metrics(ranked.valid_ids.tolist(), labels, k)
            reranked_metrics.append(rm)
            entry.update(
                reranked_metrics=asdict(rm),
                reranked_latency_ms=ranked.latency_ms,
                reranked_ids=[str(i) for i in ranked.valid_ids],
            )
        rows.append(entry)

    def mean(values: list[RelevanceMetrics]) -> dict[str, float]:
        return {
            name: sum(getattr(v, name) for v in values) / len(values)
            for name in ("recall", "mrr", "ndcg")
        }

    if progress is not None:
        progress(1.0, "Evaluation complete")
    return {
        "k": k,
        "metric": li.metric.value,
        "params": base.params,
        "n_queries": len(q),
        "ndcg_gain": "linear",
        "unjudged": "non-relevant",
        "metrics": mean(base_metrics),
        "reranked_metrics": mean(reranked_metrics) if reranked_metrics else None,
        "candidates": candidates,
        "queries": rows,
    }
