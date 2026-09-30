import faiss
import numpy as np
import pytest

from faissight.core import ivf
from faissight.core import search as S
from faissight.core import vectors as V
from faissight.core.loader import load_index
from faissight.core.search import MissReason
from tests.conftest import SMALL

D, NLIST = SMALL["d"], SMALL["nlist"]


def _hand_built():
    """2-D IVF with fixed centroids where the true nearest neighbour sits in the 2nd cell.

    Centroids c0=(0,0), c1=(10,0), c2=(0,50). The query (4.9, 0) is closest to c0, but its
    nearest stored point p0=(5.1, 0) lies just across the boundary, in c1's cell.
    """
    quantizer = faiss.IndexFlatL2(2)
    quantizer.add(np.array([[0, 0], [10, 0], [0, 50]], dtype=np.float32))
    index = faiss.IndexIVFFlat(quantizer, 2, 3)
    points = np.array(
        [[5.1, 0], [-1, 0], [0, 1], [11, 0], [0, 49]], dtype=np.float32
    )  # p0..p4 -> cells 1, 0, 0, 1, 2
    index.add(points)
    return index, points, np.array([4.9, 0], dtype=np.float32)


def _explain(li, points, q, k, nprobe):
    gt = S.GroundTruth(V.from_arrays(li, points), li.metric)
    rq = S.resolve_query(li, vector=q)
    return S.explain_query(
        li, rq, k, nprobe=nprobe, ground_truth=gt, assignments=ivf.assignments(li)
    )


def test_hand_built_cell_not_probed() -> None:
    index, points, q = _hand_built()
    li = load_index(index)
    report = _explain(li, points, q, k=2, nprobe=1)
    trace = report.ivf_trace

    np.testing.assert_array_equal(trace.probe_order, [0, 1, 2])
    np.testing.assert_allclose(
        trace.centroid_distances, [4.9**2, 5.1**2, 4.9**2 + 50**2], rtol=1e-5
    )
    np.testing.assert_array_equal(trace.probed_lists, [0])
    np.testing.assert_array_equal(report.truth.ids, [0, 2])  # p0 (d=.2), p2 (d=~5.0)
    np.testing.assert_array_equal(report.result.ids, [2, 1])  # only cell 0 searched
    np.testing.assert_array_equal(trace.result_list_nos, [0, 0])

    p0, p2 = trace.neighbours
    assert (p0.id, p0.list_no, p0.probe_rank, p0.reason, p0.found_rank) == (
        0,
        1,
        1,
        MissReason.CELL_NOT_PROBED,
        None,
    )
    assert (p2.id, p2.list_no, p2.probe_rank, p2.reason, p2.found_rank) == (
        2,
        0,
        0,
        MissReason.FOUND,
        0,
    )
    assert trace.min_nprobe_for_all == 2
    assert report.recall == 0.5
    assert trace.reason_counts()[MissReason.CELL_NOT_PROBED] == 1


def test_hand_built_min_nprobe_recovers_all() -> None:
    index, points, q = _hand_built()
    li = load_index(index)
    report = _explain(li, points, q, k=2, nprobe=2)
    assert report.recall == 1.0
    assert all(n.reason is MissReason.FOUND for n in report.ivf_trace.neighbours)


@pytest.mark.parametrize("name", ["ivf_flat", "ivf_pq", "ivf_sq8", "pca_ivf_flat", "ivf_flat_ip"])
@pytest.mark.parametrize("nprobe", [1, 3])
def test_trace_invariants(synthetic, name, nprobe) -> None:
    li = load_index(synthetic[name])
    vec_key = "vectors_normalized" if name == "ivf_flat_ip" else "vectors"
    gt = S.GroundTruth(V.from_arrays(li, np.load(synthetic[vec_key])), li.metric)
    a = ivf.assignments(li)
    queries = np.random.default_rng(3).normal(scale=4.0, size=(15, D)).astype(np.float32)
    for q in queries:
        rep = S.explain_query(
            li, S.resolve_query(li, vector=q), 10, nprobe=nprobe, ground_truth=gt, assignments=a
        )
        t = rep.ivf_trace
        assert sorted(t.probe_order.tolist()) == list(range(NLIST))
        # Results can only come from probed cells.
        assert set(t.result_list_nos.tolist()) <= set(t.probed_lists.tolist())
        for n in t.neighbours:
            if n.reason is MissReason.CELL_NOT_PROBED:
                assert n.probe_rank >= nprobe
            else:
                assert n.probe_rank < nprobe
            assert (n.found_rank is not None) == (n.reason is MissReason.FOUND)
        allowed_probed_miss = {
            "ivf_pq": MissReason.QUANTIZATION,
            "ivf_sq8": MissReason.QUANTIZATION,
            "pca_ivf_flat": MissReason.TRANSFORM,
        }.get(name, MissReason.RANKED_OUT)
        for reason in (MissReason.QUANTIZATION, MissReason.TRANSFORM, MissReason.RANKED_OUT):
            if reason is not allowed_probed_miss:
                assert t.reason_counts()[reason] == 0
        assert rep.recall == pytest.approx(t.reason_counts()[MissReason.FOUND] / 10)


def test_min_nprobe_gives_full_recall_on_ivf_flat(synthetic) -> None:
    li = load_index(synthetic["ivf_flat"])
    gt = S.GroundTruth(V.from_arrays(li, np.load(synthetic["vectors"])), li.metric)
    a = ivf.assignments(li)
    queries = np.random.default_rng(4).normal(scale=4.0, size=(20, D)).astype(np.float32)
    for q in queries:
        rq = S.resolve_query(li, vector=q)
        rep = S.explain_query(li, rq, 10, nprobe=1, ground_truth=gt, assignments=a)
        m = rep.ivf_trace.min_nprobe_for_all
        assert S.explain_query(li, rq, 10, nprobe=m, ground_truth=gt, assignments=a).recall == 1.0
        if m > 1:
            assert (
                S.explain_query(li, rq, 10, nprobe=m - 1, ground_truth=gt, assignments=a).recall
                < 1.0
            )


def test_pq_full_probe_misses_are_quantization(synthetic) -> None:
    li = load_index(synthetic["ivf_pq"])
    gt = S.GroundTruth(V.from_arrays(li, np.load(synthetic["vectors"])), li.metric)
    a = ivf.assignments(li)
    counts = dict.fromkeys(MissReason, 0)
    for q in np.load(synthetic["queries"]):
        rep = S.explain_query(
            li, S.resolve_query(li, vector=q), 10, nprobe=NLIST, ground_truth=gt, assignments=a
        )
        for reason, c in rep.ivf_trace.reason_counts().items():
            counts[reason] += c
    assert counts[MissReason.CELL_NOT_PROBED] == 0
    assert counts[MissReason.QUANTIZATION] > 0


def test_query_by_id_trace_excludes_self(synthetic) -> None:
    li = load_index(synthetic["ivf_flat"])
    src = V.from_arrays(li, np.load(synthetic["vectors"]))
    rq = S.resolve_query(li, id=42, source=src)
    rep = S.explain_query(
        li,
        rq,
        10,
        nprobe=NLIST,
        ground_truth=S.GroundTruth(src, li.metric),
        assignments=ivf.assignments(li),
    )
    assert 42 not in rep.result.ids
    assert 42 not in rep.truth.ids
    assert rep.recall == 1.0


def test_explain_without_ground_truth(synthetic) -> None:
    li = load_index(synthetic["hnsw_flat"])
    rep = S.explain_query(li, S.resolve_query(li, vector=np.zeros(D)), 5, ef_search=32)
    assert rep.truth is None
    assert rep.recall is None
    assert rep.ivf_trace is None


def test_explain_hnsw_has_recall_but_no_ivf_trace(synthetic) -> None:
    li = load_index(synthetic["hnsw_flat"])
    gt = S.GroundTruth(V.reconstruct_all(li), li.metric)
    rep = S.explain_query(li, S.resolve_query(li, vector=np.zeros(D)), 5, ground_truth=gt)
    assert rep.truth_reconstructed
    assert rep.recall is not None
    assert rep.ivf_trace is None


def test_trace_needs_ivf(synthetic) -> None:
    li = load_index(synthetic["flat_l2"])
    r = S.search(li, np.zeros(D), 5)
    with pytest.raises(ValueError, match="IVF"):
        S.trace_ivf(li, np.zeros(D), r, r, None)


def test_pca_full_probe_misses_are_transform(synthetic) -> None:
    li = load_index(synthetic["pca_ivf_flat"])
    gt = S.GroundTruth(V.from_arrays(li, np.load(synthetic["vectors"])), li.metric)
    a = ivf.assignments(li)
    counts = dict.fromkeys(MissReason, 0)
    for q in np.random.default_rng(6).normal(scale=4.0, size=(30, D)).astype(np.float32):
        rep = S.explain_query(
            li, S.resolve_query(li, vector=q), 10, nprobe=NLIST, ground_truth=gt, assignments=a
        )
        for reason, c in rep.ivf_trace.reason_counts().items():
            counts[reason] += c
    assert counts[MissReason.TRANSFORM] > 0
    assert counts[MissReason.CELL_NOT_PROBED] == counts[MissReason.RANKED_OUT] == 0


# --- probe ranks and coverage over a query set ---------------------------------------------


def test_hand_built_truth_probe_ranks() -> None:
    index, _, q = _hand_built()
    li = load_index(index)
    assign = ivf.assignments(li)
    # Neighbours p0 (cell 1), p1 (cell 0); -1 padding; 99 is not stored.
    ranks = ivf.truth_probe_ranks(li, q[None], np.array([[0, 1, -1, 99]]), assign)
    np.testing.assert_array_equal(ranks, [[1, 0, -1, 3]])


@pytest.mark.parametrize("name", ["ivf_flat", "ivf_pq", "pca_ivf_flat", "ivf_flat_ip"])
def test_truth_probe_ranks_match_trace(synthetic, name, monkeypatch) -> None:
    li = load_index(synthetic[name])
    src = V.from_arrays(li, np.load(synthetic["vectors"]))
    gt = S.GroundTruth(src, li.metric)
    assign = ivf.assignments(li)
    queries = np.load(synthetic["queries"])[:12]
    truth = gt.search_batch(queries, 10)
    # Several chunks: the result must not depend on how queries are batched.
    monkeypatch.setattr(ivf, "_PROBE_ORDER_CELLS", NLIST * 5)
    ranks = ivf.truth_probe_ranks(li, queries, truth, assign)
    for i, qv in enumerate(queries):
        report = S.explain_query(
            li, S.resolve_query(li, vector=qv), 10, nprobe=1, ground_truth=gt, assignments=assign
        )
        assert report.ivf_trace is not None
        by_id = {n.id: n.probe_rank for n in report.ivf_trace.neighbours}
        assert [by_id[int(t)] for t in truth[i]] == ranks[i].tolist()


def test_truth_probe_ranks_shape_check(synthetic) -> None:
    li = load_index(synthetic["ivf_flat"])
    with pytest.raises(ValueError, match="truth must have shape"):
        ivf.truth_probe_ranks(li, np.zeros((2, D), np.float32), np.zeros(3), ivf.assignments(li))


def test_probe_coverage_curve() -> None:
    # Query 0: neighbours at ranks 0 and 2. Query 1: rank 1 plus padding. Query 2: no truth.
    ranks = np.array([[0, 2], [1, -1], [-1, -1]])
    curve = ivf.probe_coverage(ranks, nlist=4)
    expected = [1 / 3, 1 / 3 + 1 / 6, 1 / 3 + 1 / 6 + 1 / 3, 1.0, 1.0]
    np.testing.assert_allclose(curve, expected)
    np.testing.assert_allclose(ivf.query_probe_coverage(ranks, 2), [0.5, 1.0, 1.0])
    # Unstored neighbours (rank nlist) are never covered.
    assert ivf.probe_coverage(np.array([[0, 4]]), nlist=4)[-1] == pytest.approx(0.5)
    with pytest.raises(ValueError, match="ranks must have shape"):
        ivf.probe_coverage(np.empty((0, 2), np.int64), nlist=4)
