import itertools

import faiss
import numpy as np
import pytest

from faissight.core import hnsw as H
from faissight.core import hnsw_trace as T
from faissight.core.loader import load_index
from faissight.core.types import Metric
from tests.conftest import SMALL

D = SMALL["d"]


def _overlap(li, queries, ef, k=10) -> float:
    g, x = H.extract_graph(li), T.storage_vectors(li)
    scores = []
    for q in queries:
        tr = T.trace_for_index(li, g, x, q, k, ef)
        _, faiss_ids = li.index.search(q[None], k, params=faiss.SearchParametersHNSW(efSearch=ef))
        scores.append(len(set(li.user_ids(tr.results).tolist()) & set(faiss_ids[0].tolist())) / k)
    return float(np.mean(scores))


@pytest.fixture(scope="module")
def hnsw(synthetic):
    li = load_index(synthetic["hnsw_flat"])
    return li, H.extract_graph(li), T.storage_vectors(li)


@pytest.fixture(scope="module")
def hard_queries():
    # Off-blob queries straddle clusters, where greedy search has real choices to make.
    return np.random.default_rng(3).normal(scale=4.0, size=(100, D)).astype(np.float32)


def test_trace_matches_faiss_at_ef64(hnsw, synthetic, hard_queries) -> None:
    # Acceptance: >= 95% average overlap with FAISS's own results over 100 queries.
    li, _, _ = hnsw
    queries = np.concatenate([np.load(synthetic["queries"]), hard_queries])[:100]
    assert _overlap(li, queries, ef=64) >= 0.95


@pytest.mark.parametrize("ef", [4, 8, 16, 32])
def test_trace_matches_faiss_across_ef(hnsw, hard_queries, ef) -> None:
    li, _, _ = hnsw
    assert _overlap(li, hard_queries[:50], ef=ef) >= 0.95


def _uniform(n=3000, d=24, seed=5, normalize=False):
    x = np.random.default_rng(seed).standard_normal((n, d)).astype(np.float32)
    if normalize:
        x /= np.linalg.norm(x, axis=1, keepdims=True)
    return x


def test_trace_matches_faiss_inner_product() -> None:
    x = _uniform(normalize=True)
    index = faiss.IndexHNSWFlat(24, 12, faiss.METRIC_INNER_PRODUCT)
    index.add(x)
    li = load_index(index)
    assert li.metric is Metric.IP
    assert _overlap(li, _uniform(60, seed=9, normalize=True), ef=16) >= 0.95


def test_trace_matches_faiss_through_idmap() -> None:
    x = _uniform()
    index = faiss.IndexIDMap(faiss.IndexHNSWFlat(24, 12))
    index.add_with_ids(x, np.arange(len(x), dtype=np.int64) * 5 + 2)
    assert _overlap(load_index(index), _uniform(60, seed=9), ef=16) >= 0.95


def test_trace_sq_storage_close_to_faiss() -> None:
    # SQ storage: the trace uses decoded vectors, FAISS uses its SQ distance computer.
    x = _uniform()
    index = faiss.IndexHNSWSQ(24, faiss.ScalarQuantizer.QT_8bit, 12)
    index.train(x)
    index.add(x)
    assert _overlap(load_index(index), _uniform(60, seed=9), ef=32) >= 0.9


def test_trace_structure(hnsw, hard_queries) -> None:
    li, g, x = hnsw
    tr = T.trace_for_index(li, g, x, hard_queries[0], 10, 32)
    levels = [s.level for s in tr.steps]
    assert levels == sorted(levels, reverse=True)  # top level first, level 0 last
    assert tr.level_entries[g.max_level] == g.entry_point
    assert set(tr.level_entries) == set(range(g.max_level + 1))
    # Greedy descent: each upper-level step starts where the previous one ended.
    for level in range(g.max_level, 0, -1):
        steps = [s for s in tr.steps if s.level == level]
        assert steps[0].expanded == tr.level_entries[level]
        for a, b in itertools.pairwise(steps):
            improved = [v for v in a.visits if v.accepted]
            assert b.expanded == improved[-1].node
        nxt = tr.level_entries[level - 1]
        assert nxt == steps[-1].expanded
    # Results: best first, with exact distances.
    assert len(tr.results) == 10
    assert (np.diff(tr.result_distances) >= 0).all()
    exact = np.square(x[tr.results] - li.to_core_space(hard_queries[:1])[0]).sum(1)
    np.testing.assert_allclose(tr.result_distances, exact, rtol=1e-4, atol=1e-3)
    assert set(tr.results.tolist()) <= tr.visited(0)
    assert len(tr.expansions(0)) == sum(s.level == 0 for s in tr.steps)


def test_higher_ef_visits_more(hnsw, hard_queries) -> None:
    li, g, x = hnsw
    small = T.trace_for_index(li, g, x, hard_queries[1], 10, 8)
    big = T.trace_for_index(li, g, x, hard_queries[1], 10, 128)
    assert len(big.visited(0)) > len(small.visited(0))
    s = T.summarize(big)
    assert s["ef"] == 128
    assert s["levels"][0]["distance_computations"] == sum(
        len(st.visits) for st in big.steps if st.level == 0
    )


def test_neighbour_outcomes(hnsw, synthetic, hard_queries) -> None:
    li, g, x = hnsw
    from faissight.core import search as S
    from faissight.core import vectors as V

    gt = S.GroundTruth(V.from_arrays(li, np.load(synthetic["vectors"])), li.metric)
    seen = set()
    for q in hard_queries[:40]:
        tr = T.trace_for_index(li, g, x, q, 10, 4)
        truth = gt.search(q, 10).ids
        outcomes = T.neighbour_outcomes(tr, T.internal_ids(li, truth))
        for t, o in zip(truth, outcomes, strict=True):
            if o is T.NeighbourOutcome.FOUND:
                assert t in li.user_ids(tr.results)
            elif o is T.NeighbourOutcome.NOT_REACHED:
                assert t not in tr.visited(0)
        seen.update(outcomes)
    # FAISS keeps the top-k over every evaluated node, so on exact (Flat) storage a true
    # neighbour whose distance was computed is always returned: misses are all NOT_REACHED.
    assert seen == {T.NeighbourOutcome.FOUND, T.NeighbourOutcome.NOT_REACHED}


def test_internal_ids_roundtrip() -> None:
    x = _uniform(500)
    index = faiss.IndexIDMap(faiss.IndexHNSWFlat(24, 8))
    ids = np.arange(500, dtype=np.int64)[::-1] * 3 + 11
    index.add_with_ids(x, ids)
    li = load_index(index)
    internal = T.internal_ids(li, [ids[7], ids[0], 999_999])
    np.testing.assert_array_equal(internal, [7, 0, -1])
    np.testing.assert_array_equal(li.user_ids(internal[:2]), [ids[7], ids[0]])


def test_internal_ids_without_idmap(hnsw) -> None:
    li, _, _ = hnsw
    np.testing.assert_array_equal(T.internal_ids(li, [0, 5, -3, SMALL["n"]]), [0, 5, -1, -1])


def test_minimax_heap_matches_faiss_semantics() -> None:
    h = T._MinimaxHeap(3)
    assert h.push(1, 5.0)
    assert h.push(2, 3.0)
    assert h.push(3, 4.0)
    assert not h.push(4, 6.0)  # full and not better than the worst
    assert h.pop_min() == (2, 3.0)
    assert h.size() == 2
    # The popped slot still takes capacity: pushing evicts the max (5.0), not the hole.
    assert h.push(5, 1.0)
    assert h.count_below(10.0) == 2
    assert sorted(h.pop_min() for _ in range(h.size())) == [(3, 4.0), (5, 1.0)]


def test_trace_errors(hnsw) -> None:
    _, g, x = hnsw
    with pytest.raises(ValueError, match=">= 1"):
        T.trace_search(g, x, np.zeros(D), 0, 16)


def test_trace_empty_graph_is_a_clear_error() -> None:
    li = load_index(faiss.IndexHNSWFlat(4, 8))
    g = H.extract_graph(li)
    with pytest.raises(ValueError, match="empty"):
        T.trace_search(g, np.empty((0, 4), dtype=np.float32), np.zeros(4), 5, 16)


# --- node vectors without copying the storage ---------------------------------------------


@pytest.mark.parametrize("spec", ["HNSW8", "HNSW8_SQ8", "HNSW8_PQ4"])
def test_node_vectors_match_storage_vectors(spec) -> None:
    x = np.random.default_rng(0).standard_normal((300, 16)).astype(np.float32)
    index = faiss.index_factory(16, spec)
    index.train(x)
    index.add(x)
    li = load_index(index)
    full = T.storage_vectors(li)
    lazy = T.node_vectors(li)
    assert len(lazy) == len(full)
    nodes = np.array([5, 0, 299, 5])
    np.testing.assert_array_equal(np.asarray(lazy[nodes]), full[nodes])
    # Flat storage is a view of the index's own memory, not a copy.
    assert isinstance(lazy, np.ndarray) == (spec == "HNSW8")
    g = H.extract_graph(li)
    a = T.trace_search(g, lazy, x[7] + 0.1, 5, 16)
    b = T.trace_search(g, full, x[7] + 0.1, 5, 16)
    np.testing.assert_array_equal(a.results, b.results)
