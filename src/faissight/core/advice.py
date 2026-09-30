"""Suggested next steps from a sweep: what to change, and the measurements behind it.

Each rule fires only on evidence the sweep measured. IVF sweeps with probe coverage (see
:attr:`SweepPoint.probe_coverage`) can tell "probe more lists" apart from "the ranking
inside probed lists loses neighbours"; HNSW sweeps only have the recall trend.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Literal

from faissight.core.ivf import ListStats
from faissight.core.search import MissReason, probed_miss_reason
from faissight.core.sweep import SweepParam, SweepPoint, SweepResult
from faissight.core.types import IndexKind, LoadedIndex

# Recall gained per doubling of the parameter below which a curve counts as flat.
PLATEAU_GAIN_PER_DOUBLING = 0.005
# Doublings of the parameter a still-rising curve may need before "try larger values" is
# no longer useful advice (8x the largest value tried).
MAX_DOUBLINGS_TO_TARGET = 3
# Share of true neighbours lost inside probed lists worth calling out.
MIN_RANKING_LOSS = 0.005
# A query "fails" when it finds fewer than half of its true neighbours...
FAILING_RECALL = 0.5
# ...and the tail is worth a suggestion when at least this share of queries fails.
MIN_FAILING_SHARE = 0.02
# FAISS imbalance factor (1.0 = even) and empty-list share that count as uneven lists.
IMBALANCE_FACTOR = 2.0
EMPTY_LIST_SHARE = 0.1
# Values offered for a follow-up sweep.
MAX_SUGGESTED_VALUES = 8

View = Literal["overview", "quantization", "compare", "query"]


class SuggestionKind(str, Enum):
    PROBE_MORE = "PROBE_MORE"
    """IVF: true neighbours sit in lists the search didn't probe."""
    RANKING_LIMIT = "RANKING_LIMIT"
    """IVF: neighbours in probed lists are ranked out, so more probing can't reach the target."""
    SEARCH_WIDER = "SEARCH_WIDER"
    """Recall is still rising at the largest value tried."""
    RECALL_PLATEAU = "RECALL_PLATEAU"
    """Recall has levelled off below the target."""
    FAILING_QUERIES = "FAILING_QUERIES"
    """The mean meets the target but a tail of queries finds few of its neighbours."""
    LIST_IMBALANCE = "LIST_IMBALANCE"
    """IVF lists are unevenly filled."""


@dataclass(frozen=True)
class Evidence:
    """One measurement behind a suggestion, already formatted for display."""

    label: str
    value: str


@dataclass(frozen=True)
class Suggestion:
    """A next step, why it is suggested, and optionally where to take it."""

    kind: SuggestionKind
    title: str
    detail: str
    evidence: list[Evidence] = field(default_factory=list)
    sweep_values: list[int] | None = None
    """Values for a follow-up sweep of the same parameter."""
    view: View | None = None
    """A view that shows more (``query`` together with ``query_id``)."""
    query_id: int | None = None
    at_value: int | None = None
    """The parameter value the evidence was measured at."""


def advise(
    result: SweepResult,
    li: LoadedIndex,
    target_recall: float,
    *,
    confident: bool = False,
    list_stats: ListStats | None = None,
    max_value: int | None = None,
    can_compare: bool = False,
) -> list[Suggestion]:
    """Suggested next steps for a finished sweep, most important first.

    ``confident`` uses the same rule as :meth:`SweepResult.recommend`. ``max_value`` is the
    largest value the parameter accepts (nlist for nprobe; ``None`` = unbounded) and
    ``can_compare`` says whether other indexes are loaded for a side-by-side comparison.
    """
    if not result.points:
        return []
    if result.param is SweepParam.NPROBE and max_value is None:
        max_value = int(li.ivf.nlist)
    out: list[Suggestion] = []
    rec = result.recommend(target_recall, confident=confident)
    top = max(result.points, key=lambda p: p.value)
    if rec is None:
        if top.probe_coverage is not None and result.coverage_curve is not None:
            out += _ivf_short(result, li, target_recall, top, can_compare)
        else:
            out += _trend_short(result, li, target_recall, top, max_value, can_compare)
    else:
        # A low target already accepts queries that find few neighbours.
        tail = _failing_queries(result, rec) if target_recall > FAILING_RECALL else None
        if tail is not None:
            out.append(tail)
    if list_stats is not None:
        uneven = _imbalance(list_stats, rec or top, result.param)
        if uneven is not None:
            out.append(uneven)
    return out


def _pct(x: float) -> str:
    return f"{x:.0%}" if x >= 0.1 or x == 0 else f"{x:.1%}"


def _ladder(start: int, end: int, limit: int | None) -> list[int]:
    """``start``, powers of two up to ``end``, then ``end`` and one doubling past it."""
    if limit is not None:
        end = min(end, limit)
    values = {start, end}
    v = 1 << max(start, 1).bit_length()
    while v < end:
        values.add(v)
        v <<= 1
    past = end * 2 if limit is None else min(end * 2, limit)
    values.add(past)
    ordered = sorted(values)
    if len(ordered) > MAX_SUGGESTED_VALUES:
        # Keep both ends and thin out the middle.
        step = (len(ordered) - 1) / (MAX_SUGGESTED_VALUES - 1)
        ordered = sorted({ordered[round(i * step)] for i in range(MAX_SUGGESTED_VALUES)})
    return ordered


def _ivf_short(
    result: SweepResult, li: LoadedIndex, target: float, top: SweepPoint, can_compare: bool
) -> list[Suggestion]:
    """IVF misses the target: split the loss into unprobed lists vs ranking in probed ones."""
    assert top.probe_coverage is not None
    assert result.coverage_curve is not None
    nlist = len(result.coverage_curve) - 1
    coverage, recall = top.probe_coverage, top.recall
    unprobed, ranked_out = 1.0 - coverage, max(coverage - recall, 0.0)
    # Share of true neighbours in probed lists that the search returned.
    kept = recall / coverage if coverage > 0 else 1.0
    out: list[Suggestion] = []
    needed = result.nprobe_for_coverage(target)
    if coverage < target - 1e-9 and needed is not None and needed > top.value:
        detail = (
            f"At nprobe {top.value}, {_pct(unprobed)} of the true neighbours are in lists the "
            f"search didn't probe, so recall can't exceed {coverage:.3f}. Probing {needed} of "
            f"{nlist} lists puts {target:.2f} of them within reach."
        )
        if ranked_out >= MIN_RANKING_LOSS:
            detail += (
                f" Expect less than that: {_pct(1 - kept)} of the neighbours in probed lists "
                "are still ranked out."
            )
        out.append(
            Suggestion(
                SuggestionKind.PROBE_MORE,
                f"Probe more lists: try nprobe up to {needed}",
                detail,
                [
                    Evidence(f"In probed lists at nprobe {top.value}", f"{coverage:.3f}"),
                    Evidence(f"Recall at nprobe {top.value}", f"{recall:.3f}"),
                    Evidence(f"nprobe with {target:.2f} in probed lists", f"{needed} of {nlist}"),
                ],
                sweep_values=_ladder(top.value, needed, nlist),
                at_value=top.value,
            )
        )
    if kept < target - 1e-9 and ranked_out >= MIN_RANKING_LOSS:
        # The binding limit: probing more lists can't reach the target on its own.
        out.insert(0, _ranking_limit(result, li, top, ranked_out, kept, can_compare))
    return out


def _ranking_limit(
    result: SweepResult,
    li: LoadedIndex,
    top: SweepPoint,
    ranked_out: float,
    kept: float,
    can_compare: bool,
) -> Suggestion:
    reason = probed_miss_reason(li)
    lost = (
        f"{_pct(ranked_out)} of the true neighbours were in probed lists but ranked out of "
        f"the top {result.k}, and only {_pct(kept)} of those in probed lists were returned: "
        f"probing every list would leave recall near {kept:.2f}."
    )
    view: View = "compare" if can_compare else "quantization"
    if reason is MissReason.QUANTIZATION:
        title = "Compression is limiting recall"
        if li.has_refine:
            fix = (
                f"Raise the refine step's k_factor (now {li.refine_k_factor:g}) so more "
                "candidates are re-ranked with exact distances."
            )
        else:
            fix = (
                "Re-rank candidates with exact distances (IndexRefineFlat, or ',RFlat' in "
                "index_factory), or store larger codes (more PQ sub-quantizers, or SQ8)."
            )
        detail = f"{lost} {fix}"
        if result.truth_reconstructed:
            detail += " Ground truth uses decoded vectors, so the real loss is larger."
    elif reason is MissReason.TRANSFORM:
        title = "Dimensionality reduction is limiting recall"
        steps = ", ".join(f"{t.name} {t.d_in}→{t.d_out}" for t in li.transforms)
        detail = (
            f"{lost} The index compares vectors after {steps}, which reorders close "
            "neighbours. Keep more dimensions, or re-rank candidates in the original space."
        )
        view = "compare" if can_compare else "query"
    else:
        title = "Neighbours are ranked out of probed lists"
        detail = (
            f"{lost} IVF-Flat distances are exact, so this usually means the raw vectors "
            "differ from what was indexed (normalisation, ids), or many distances tie."
        )
        view = "query"
    worst = top.worst_queries[0] if top.worst_queries else None
    return Suggestion(
        SuggestionKind.RANKING_LIMIT,
        title,
        detail,
        [
            Evidence(f"Ranked out of probed lists at nprobe {top.value}", f"{ranked_out:.3f}"),
            Evidence("Returned from probed lists", _pct(kept)),
        ],
        view=view,
        query_id=worst.id if view == "query" and worst is not None else None,
        at_value=top.value,
    )


def _trend_short(
    result: SweepResult,
    li: LoadedIndex,
    target: float,
    top: SweepPoint,
    max_value: int | None,
    can_compare: bool,
) -> list[Suggestion]:
    """No coverage data (HNSW): judge by whether recall still rises at the largest values."""
    name = result.param.value
    can_grow = max_value is None or top.value < max_value
    lower = [p for p in result.points if p.value < top.value]
    if not lower:
        if not can_grow:
            return []
        return [
            Suggestion(
                SuggestionKind.SEARCH_WIDER,
                f"Try larger {name} values",
                f"Only {name} {top.value} was measured (recall {top.recall:.3f}); sweep larger "
                "values to see how far recall rises.",
                [Evidence(f"Recall at {name} {top.value}", f"{top.recall:.3f}")],
                sweep_values=_ladder(top.value, top.value * 8, max_value),
                at_value=top.value,
            )
        ]
    prev = max(lower, key=lambda p: p.value)
    gain = top.recall - prev.recall
    per_doubling = gain / max(math.log2(top.value / prev.value), 1e-9)
    trend = [
        Evidence(f"Recall at {name} {prev.value}", f"{prev.recall:.3f}"),
        Evidence(f"Recall at {name} {top.value}", f"{top.recall:.3f}"),
        Evidence("Gain per doubling", f"{per_doubling:+.3f}"),
    ]
    rising = per_doubling >= PLATEAU_GAIN_PER_DOUBLING
    # At the current rate, how many more doublings would reach the target (a rough guide:
    # recall gains usually shrink as the parameter grows, so this is optimistic).
    doublings = (target - top.recall) / per_doubling if rising else math.inf
    if rising and can_grow and doublings <= MAX_DOUBLINGS_TO_TARGET:
        return [
            Suggestion(
                SuggestionKind.SEARCH_WIDER,
                f"Recall is still rising: try larger {name} values",
                f"Recall went from {prev.recall:.3f} to {top.recall:.3f} between {name} "
                f"{prev.value} and {top.value}, {target - top.recall:.3f} short of the target. "
                "Larger values cost latency; the sweep shows how much.",
                trend,
                sweep_values=_ladder(top.value, top.value * 8, max_value),
                at_value=top.value,
            )
        ]
    if rising and not can_grow:
        why = f"{name} is already at its largest allowed value ({max_value})."
    elif rising:
        why = (
            f"Recall is rising too slowly to reach the target: at {per_doubling:+.3f} per "
            f"doubling it would need about {math.ceil(doublings)} more doublings of {name}."
        )
    else:
        why = f"Raising {name} has stopped helping."
    p = li.params
    if li.kind.is_hnsw:
        build = " or ".join(
            f"{label} (now {v})"
            for label, v in (("M", p.hnsw_m), ("efConstruction", p.ef_construction))
            if v is not None
        )
        graph = f"rebuild with a larger {build}" if build else "rebuild with a denser graph"
        if li.kind is IndexKind.HNSW_FLAT:
            fix = f"The graph limits recall: {graph}."
        else:
            fix = f"The graph or the compressed codes limit recall: {graph}, or store larger codes."
    else:
        fix = "Something other than this parameter limits recall; compare other index settings."
    return [
        Suggestion(
            SuggestionKind.RECALL_PLATEAU,
            (
                f"Recall levels off at {top.recall:.3f}, below the target"
                if not rising
                else f"More {name} won't reach the target"
            ),
            f"{why} {fix}",
            trend,
            view="compare" if can_compare else None,
            at_value=top.value,
        )
    ]


def _failing_queries(result: SweepResult, rec: SweepPoint) -> Suggestion | None:
    total = sum(n for _, n in rec.recall_distribution)
    failing = sum(n for r, n in rec.recall_distribution if r < FAILING_RECALL - 1e-9)
    if total == 0 or failing == 0 or failing / total < MIN_FAILING_SHARE:
        return None
    name = result.param.value
    detail = (
        f"At {name} {rec.value}, {failing} of {total} queries ({_pct(failing / total)}) return "
        f"fewer than half of their {result.k} true neighbours."
    )
    evidence = [
        Evidence(f"Queries below recall {FAILING_RECALL:g}", f"{failing} of {total}"),
        Evidence("Worst query's recall", f"{min(r for r, _ in rec.recall_distribution):.2f}"),
    ]
    bad = [w for w in rec.worst_queries if w.recall < FAILING_RECALL - 1e-9]
    missed = sum(1 - w.recall for w in bad)
    if bad and missed > 0 and all(w.probe_coverage is not None for w in bad):
        unprobed = sum(1 - (w.probe_coverage or 0.0) for w in bad) / missed
        evidence.append(Evidence("Their misses in unprobed lists", _pct(unprobed)))
        if unprobed >= 0.5:
            detail += (
                f" Among the worst, {_pct(unprobed)} of missed neighbours are in lists that "
                "weren't probed: these queries fall between clusters, and a larger nprobe "
                "fixes them at a latency cost. Check they resemble real traffic first."
            )
        else:
            detail += (
                f" Among the worst, {_pct(1 - unprobed)} of missed neighbours were in probed "
                "lists but ranked out, so a larger nprobe won't fix them."
            )
    else:
        detail += " Open the worst one to see why its neighbours were missed."
    worst = rec.worst_queries[0] if rec.worst_queries else None
    return Suggestion(
        SuggestionKind.FAILING_QUERIES,
        "Some queries fail although the mean meets the target",
        detail,
        evidence,
        view="query" if worst is not None and worst.id is not None else None,
        at_value=rec.value,
        query_id=worst.id if worst is not None else None,
    )


def _imbalance(stats: ListStats, at: SweepPoint, param: SweepParam) -> Suggestion | None:
    nlist = len(stats.sizes)
    empty_share = stats.n_empty / nlist if nlist else 0.0
    if stats.imbalance_factor < IMBALANCE_FACTOR and empty_share < EMPTY_LIST_SHARE:
        return None
    evidence = [
        Evidence("Imbalance factor (1.0 = even)", f"{stats.imbalance_factor:.2f}"),
        Evidence("Vectors in the largest 5% of lists", _pct(stats.top_5pct_share)),
        Evidence("Largest / median list", f"{stats.max} / {stats.median:g}"),
    ]
    if stats.n_empty:
        evidence.append(Evidence("Empty lists", f"{stats.n_empty} of {nlist}"))
    if at.latency_mean_ms > 0:
        evidence.append(
            Evidence(
                f"p95 / mean latency at {param.value} {at.value}",
                f"{at.latency_p95_ms / at.latency_mean_ms:.1f}x",
            )
        )
    return Suggestion(
        SuggestionKind.LIST_IMBALANCE,
        "Inverted lists are unevenly filled",
        f"The largest 5% of lists hold {_pct(stats.top_5pct_share)} of the vectors"
        + (f" and {stats.n_empty} lists are empty" if stats.n_empty else "")
        + ". Queries that probe the big lists scan more codes, so latency varies by query, "
        "and one nprobe covers some regions far better than others. Retrain the coarse "
        "quantizer on a sample like the stored vectors, or change nlist.",
        evidence,
        view="overview",
        at_value=at.value,
    )
