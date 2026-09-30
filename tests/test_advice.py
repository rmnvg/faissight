import numpy as np
import pytest

from faissight.core import advice as A
from faissight.core import ivf
from faissight.core.advice import SuggestionKind as K
from faissight.core.loader import load_index
from faissight.core.sweep import SweepParam, SweepPoint, SweepResult, WorstQuery
from tests.conftest import SMALL

NLIST = SMALL["nlist"]


@pytest.fixture(scope="module")
def indexes(synthetic):
    return {
        name: load_index(synthetic[name])
        for name in ("ivf_flat", "ivf_pq", "pca_ivf_flat", "hnsw_flat")
    }


def _point(value, recall, coverage=None, *, distribution=None, worst=None, latency=(1.0, 2.0)):
    return SweepPoint(
        value=value,
        recall=recall,
        latency_mean_ms=latency[0],
        latency_p95_ms=latency[1],
        recall_ci_low=recall - 0.01,
        recall_ci_high=min(recall + 0.01, 1.0),
        recall_distribution=distribution or [],
        worst_queries=worst or [],
        probe_coverage=coverage,
    )


def _ivf(points, curve=None):
    return SweepResult(
        SweepParam.NPROBE,
        10,
        100,
        "sampled",
        False,
        points,
        coverage_curve=None if curve is None else np.asarray(curve, np.float64),
    )


def _hnsw(points):
    return SweepResult(SweepParam.EF_SEARCH, 10, 100, "sampled", False, points)


def _curve(**at):
    """Coverage 0..NLIST, linear between the given nprobe: coverage points."""
    known = {0: 0.0, NLIST: 1.0, **{int(k[1:]): v for k, v in at.items()}}
    xs, ys = zip(*sorted(known.items()), strict=True)
    return np.interp(np.arange(NLIST + 1), xs, ys)


def _balanced():
    sizes = np.full(NLIST, 100, dtype=np.int64)
    return ivf.ListStats(sizes, 0, 1.0, 100, 100.0, 100, [(0, 100)], 0.05)


def test_probe_more_when_neighbours_sit_in_unprobed_lists(indexes) -> None:
    curve = _curve(n2=0.6, n4=0.8, n8=0.97)
    r = _ivf([_point(1, 0.4, curve[1]), _point(2, 0.6, curve[2]), _point(4, 0.8, curve[4])], curve)
    (s,) = A.advise(r, indexes["ivf_flat"], 0.95, list_stats=_balanced())
    assert s.kind is K.PROBE_MORE
    needed = r.nprobe_for_coverage(0.95)
    assert needed is not None
    assert 4 < needed <= 8
    assert f"nprobe up to {needed}" in s.title
    assert s.sweep_values is not None
    assert s.sweep_values[0] == 4
    assert needed in s.sweep_values
    assert max(s.sweep_values) <= NLIST
    assert "ranked out" not in s.detail  # IVF-Flat: nothing lost inside probed lists


def test_compression_limit_comes_first(indexes) -> None:
    curve = _curve(n4=0.8)
    r = _ivf([_point(2, 0.3, curve[2]), _point(4, 0.5, curve[4])], curve)
    out = A.advise(r, indexes["ivf_pq"], 0.95)
    assert [s.kind for s in out] == [K.RANKING_LIMIT, K.PROBE_MORE]
    limit = out[0]
    assert limit.title == "Compression is limiting recall"
    assert "IndexRefineFlat" in limit.detail
    assert "0.62" in limit.detail  # 0.5 of 0.8 probed neighbours returned
    assert limit.view == "quantization"
    assert "ranked out" in out[1].detail
    assert A.advise(r, indexes["ivf_pq"], 0.95, can_compare=True)[0].view == "compare"


def test_transform_and_ranked_out_limits(indexes) -> None:
    curve = _curve(n16=1.0)
    worst = [WorstQuery(3, 42, 0.1, 1.0)]
    r = _ivf([_point(NLIST, 0.5, 1.0, worst=worst)], curve)
    (pca,) = A.advise(r, indexes["pca_ivf_flat"], 0.95)
    assert pca.title == "Dimensionality reduction is limiting recall"
    assert "→" in pca.detail
    assert (pca.view, pca.query_id) == ("query", 42)
    (flat,) = A.advise(r, indexes["ivf_flat"], 0.95)
    assert flat.title == "Neighbours are ranked out of probed lists"
    assert (flat.view, flat.query_id) == ("query", 42)


def test_hnsw_still_rising_suggests_larger_values(indexes) -> None:
    r = _hnsw([_point(16, 0.80), _point(32, 0.86), _point(64, 0.91)])
    (s,) = A.advise(r, indexes["hnsw_flat"], 0.95)
    assert s.kind is K.SEARCH_WIDER
    assert s.sweep_values == [64, 128, 256, 512, 1024]
    (capped,) = A.advise(r, indexes["hnsw_flat"], 0.95, max_value=200)
    assert capped.sweep_values is not None
    assert max(capped.sweep_values) == 200


def test_hnsw_plateau_slow_rise_and_cap(indexes) -> None:
    flat = _hnsw([_point(64, 0.900), _point(128, 0.902)])
    (s,) = A.advise(flat, indexes["hnsw_flat"], 0.95)
    assert s.kind is K.RECALL_PLATEAU
    assert "stopped helping" in s.detail
    assert "M (now" in s.detail
    assert "compressed" not in s.detail  # HNSW-Flat stores exact vectors
    assert s.sweep_values is None

    slow = _hnsw([_point(64, 0.50), _point(128, 0.52)])
    (s,) = A.advise(slow, indexes["hnsw_flat"], 0.95)
    assert s.kind is K.RECALL_PLATEAU
    assert s.title == "More efSearch won't reach the target"
    assert "22 more doublings" in s.detail

    rising = _hnsw([_point(64, 0.90), _point(128, 0.93)])
    (s,) = A.advise(rising, indexes["hnsw_flat"], 0.95, max_value=128)
    assert s.kind is K.RECALL_PLATEAU
    assert "largest allowed value (128)" in s.detail


def test_single_point_suggests_a_ladder(indexes) -> None:
    (s,) = A.advise(_hnsw([_point(16, 0.5)]), indexes["hnsw_flat"], 0.95)
    assert s.kind is K.SEARCH_WIDER
    assert s.sweep_values == [16, 32, 64, 128, 256]
    assert A.advise(_hnsw([_point(16, 0.5)]), indexes["hnsw_flat"], 0.95, max_value=16) == []


def test_failing_queries_behind_a_good_mean(indexes) -> None:
    worst = [WorstQuery(0, 7, 0.2, 0.3), WorstQuery(1, 8, 0.3, 0.4)]
    point = _point(8, 0.96, 0.97, distribution=[(0.2, 1), (0.3, 4), (1.0, 95)], worst=worst)
    (s,) = A.advise(_ivf([point], _curve(n8=0.97)), indexes["ivf_flat"], 0.95)
    assert s.kind is K.FAILING_QUERIES
    assert "5 of 100 queries" in s.detail
    assert "lists that weren't probed" in s.detail  # 1.3 of 1.5 misses unprobed
    assert (s.view, s.query_id) == ("query", 7)

    ranked = [WorstQuery(0, 7, 0.2, 1.0)]
    point = _point(8, 0.96, 1.0, distribution=[(0.2, 5), (1.0, 95)], worst=ranked)
    (s,) = A.advise(_ivf([point], _curve(n8=1.0)), indexes["ivf_pq"], 0.95)
    assert "won't fix them" in s.detail

    # A target at or below the failing threshold already accepts those queries.
    assert A.advise(_ivf([point], _curve(n8=1.0)), indexes["ivf_pq"], 0.5) == []

    # One failing query in 100 is under the threshold.
    point = _point(8, 0.99, 1.0, distribution=[(0.2, 1), (1.0, 99)])
    assert A.advise(_ivf([point], _curve(n8=1.0)), indexes["ivf_flat"], 0.95) == []


def test_list_imbalance(indexes) -> None:
    sizes = np.array([1000] + [10] * (NLIST - 2) + [0], dtype=np.int64)
    stats = ivf.ListStats(
        sizes, 1, ivf.imbalance_factor(sizes), 0, 10.0, 1000, [(0, 1000)], 1000 / sizes.sum()
    )
    r = _ivf([_point(8, 0.99, 1.0, latency=(1.0, 4.0))], _curve(n8=1.0))
    (s,) = A.advise(r, indexes["ivf_flat"], 0.95, list_stats=stats)
    assert s.kind is K.LIST_IMBALANCE
    assert s.view == "overview"
    labels = {e.label: e.value for e in s.evidence}
    assert labels["Empty lists"] == f"1 of {NLIST}"
    assert labels["p95 / mean latency at nprobe 8"] == "4.0x"
    assert A.advise(r, indexes["ivf_flat"], 0.95, list_stats=_balanced()) == []


def test_confident_rule_and_empty_sweep(indexes) -> None:
    r = _hnsw([_point(64, 0.955), _point(128, 0.957)])
    assert A.advise(r, indexes["hnsw_flat"], 0.95) == []
    # The interval's lower end misses the target, and recall is flat.
    (s,) = A.advise(r, indexes["hnsw_flat"], 0.95, confident=True)
    assert s.kind is K.RECALL_PLATEAU
    assert A.advise(_hnsw([]), indexes["hnsw_flat"], 0.95) == []


@pytest.mark.parametrize(
    ("start", "end", "limit", "expected"),
    [
        (4, 28, 64, [4, 8, 16, 28, 56]),
        (8, 16, 16, [8, 16]),
        (6, 40, None, [6, 8, 16, 32, 40, 80]),
    ],
)
def test_ladder(start, end, limit, expected) -> None:
    assert A._ladder(start, end, limit) == expected


def test_ladder_is_bounded() -> None:
    values = A._ladder(1, 1 << 20, None)
    assert len(values) <= A.MAX_SUGGESTED_VALUES
    assert values[0] == 1
    assert values[-1] == 1 << 21


def test_every_suggestion_says_where_it_was_measured(indexes) -> None:
    curve = _curve(n4=0.8)
    r = _ivf([_point(2, 0.3, curve[2]), _point(4, 0.5, curve[4])], curve)
    assert {s.at_value for s in A.advise(r, indexes["ivf_pq"], 0.95)} == {4}
    r = _hnsw([_point(64, 0.900), _point(128, 0.902)])
    assert [s.at_value for s in A.advise(r, indexes["hnsw_flat"], 0.95)] == [128]


def test_latency_budget_conflict_comes_first(indexes) -> None:
    r = _hnsw(
        [
            _point(16, 0.90, latency=(1.0, 2.0)),
            _point(64, 0.96, latency=(3.0, 6.0)),
            _point(128, 0.97, latency=(5.0, 9.0)),
        ]
    )
    (s,) = A.advise(r, indexes["hnsw_flat"], 0.95, max_p95_ms=4.0)
    assert s.kind is K.LATENCY_BUDGET
    assert "efSearch 64, with a p95 latency of 6 ms: over the 4 ms budget" in s.detail
    assert "the best is efSearch 16 with recall 0.900" in s.detail
    assert s.at_value == 64
    assert A.advise(r, indexes["hnsw_flat"], 0.95, max_p95_ms=6.0) == []
    (s,) = A.advise(r, indexes["hnsw_flat"], 0.95, max_p95_ms=1.0)
    assert "No setting tried is within the budget" in s.detail
