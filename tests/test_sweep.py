import itertools

import faiss
import numpy as np
import pytest

from faissight.core import search as S
from faissight.core import sweep as W
from faissight.core import vectors as V
from faissight.core.loader import load_index
from tests.conftest import SMALL

N, D, NLIST = SMALL["n"], SMALL["d"], SMALL["nlist"]
K = 10


@pytest.fixture(scope="module")
def ivf_setup(synthetic):
    li = load_index(synthetic["ivf_flat"])
    src = V.from_arrays(li, np.load(synthetic["vectors"]))
    gt = S.GroundTruth(src, li.metric)
    qs = W.sample_queries(src, 60)
    return li, src, gt, qs, W.ground_truth_ids(gt, qs, K)


# --- defaults & query sets -----------------------------------------------------------------


def test_default_values(synthetic) -> None:
    assert W.default_values(load_index(synthetic["ivf_flat"])) == [1, 2, 4, 8, 16]
    assert W.default_values(load_index(synthetic["hnsw_flat"])) == list(W.DEFAULT_EF_VALUES)
    x = np.random.default_rng(0).standard_normal((500, 8)).astype(np.float32)
    odd = faiss.IndexIVFFlat(faiss.IndexFlatL2(8), 8, 12)
    odd.train(x)
    assert W.default_values(load_index(odd)) == [1, 2, 4, 8, 12]


def test_param_errors(synthetic) -> None:
    with pytest.raises(ValueError, match="no search parameter"):
        W.default_values(load_index(synthetic["flat_l2"]))
    with pytest.raises(ValueError, match="HNSW"):
        W.default_values(load_index(synthetic["ivf_flat"]), "efSearch")
    with pytest.raises(ValueError, match="IVF"):
        W.default_values(load_index(synthetic["hnsw_flat"]), "nprobe")


def test_sample_queries(ivf_setup) -> None:
    _, src, _, qs, _ = ivf_setup
    assert len(qs) == 60
    assert qs.origin == "sampled"
    np.testing.assert_array_equal(src.get(qs.exclude_ids), qs.vectors)
    again = W.sample_queries(src, 60)
    np.testing.assert_array_equal(again.exclude_ids, qs.exclude_ids)
    assert len(W.sample_queries(src, 10 * N)) == N
    with pytest.raises(ValueError, match="n_queries"):
        W.sample_queries(src, 0)


def test_given_queries(synthetic) -> None:
    li = load_index(synthetic["ivf_flat"])
    qs = W.given_queries(li, np.load(synthetic["queries"]))
    assert qs.exclude_ids is None
    assert qs.origin == "given"
    with pytest.raises(ValueError, match="shape"):
        W.given_queries(li, np.zeros((3, D + 1)))


def test_ground_truth_ids_exclude_self(ivf_setup) -> None:
    _, _, gt, qs, truth = ivf_setup
    assert truth.shape == (60, K)
    assert not (truth == qs.exclude_ids[:, None]).any()
    np.testing.assert_array_equal(truth[0], gt.search(qs.vectors[0], K, int(qs.exclude_ids[0])).ids)


def test_search_batch_shape_check(ivf_setup) -> None:
    _, _, gt, _, _ = ivf_setup
    with pytest.raises(ValueError, match="shape"):
        gt.search_batch(np.zeros((2, D + 1)), K)


# --- sweep ---------------------------------------------------------------------------------


def test_recall_monotonic_in_nprobe(ivf_setup) -> None:
    # Acceptance: on IVFFlat, probing more cells never loses a true neighbour.
    li, _, _, qs, truth = ivf_setup
    res = W.sweep(li, qs, truth, k=K)
    assert res.param is W.SweepParam.NPROBE
    assert [p.value for p in res.points] == [1, 2, 4, 8, 16]
    recalls = [p.recall for p in res.points]
    assert all(b >= a - 1e-12 for a, b in itertools.pairwise(recalls))
    assert recalls[-1] == 1.0
    assert all(p.latency_mean_ms > 0 and p.latency_p95_ms >= 0 for p in res.points)
    assert res.n_queries == 60
    assert res.query_origin == "sampled"


def test_recommended_value_meets_target(ivf_setup) -> None:
    # Acceptance: the recommended value meets the target recall on the query set.
    li, _, _, qs, truth = ivf_setup
    res = W.sweep(li, qs, truth, k=K)
    for target in (0.5, 0.9, 0.95, 0.99, 1.0):
        rec = res.recommend(target)
        assert rec is not None
        assert rec.recall >= target
        cheaper = [p for p in res.points if p.value < rec.value]
        assert all(p.recall < target for p in cheaper)


def test_recommend_unreachable_target(ivf_setup) -> None:
    li, _, _, qs, truth = ivf_setup
    res = W.sweep(li, qs, truth, values=[1], k=K)
    assert res.recommend(1.01) is None


def test_threads_pinned_then_restored(ivf_setup) -> None:
    li, _, _, qs, truth = ivf_setup
    before = faiss.omp_get_max_threads()
    during = []
    W.sweep(
        li,
        qs,
        truth,
        values=[1, 2],
        k=K,
        progress=lambda f, m: during.append(faiss.omp_get_max_threads()),
    )
    assert set(during[:-1]) == {1}
    assert faiss.omp_get_max_threads() == before


def test_threads_restored_on_error(ivf_setup, monkeypatch) -> None:
    li, _, _, qs, truth = ivf_setup
    before = faiss.omp_get_max_threads()

    def boom(frac, msg):
        if msg.startswith("nprobe"):
            raise RuntimeError("interrupted")

    with pytest.raises(RuntimeError):
        W.sweep(li, qs, truth, values=[1], k=K, progress=boom)
    assert faiss.omp_get_max_threads() == before


def test_sweep_given_queries(synthetic) -> None:
    li = load_index(synthetic["ivf_pq"])
    gt = S.GroundTruth(V.from_arrays(li, np.load(synthetic["vectors"])), li.metric)
    qs = W.given_queries(li, np.load(synthetic["queries"]))
    res = W.sweep(li, qs, W.ground_truth_ids(gt, qs, K), values=[1, NLIST], k=K)
    assert res.query_origin == "given"
    # PQ can't reach full recall even probing everything: quantization.
    assert res.points[-1].recall < 1.0


def test_sweep_hnsw(synthetic) -> None:
    li = load_index(synthetic["hnsw_flat"])
    src = V.from_arrays(li, np.load(synthetic["vectors"]))
    qs = W.sample_queries(src, 40)
    truth = W.ground_truth_ids(S.GroundTruth(src, li.metric), qs, K)
    res = W.sweep(li, qs, truth, values=[16, 256], k=K)
    assert res.param is W.SweepParam.EF_SEARCH
    assert res.points[1].recall >= res.points[0].recall
    assert res.points[1].recall > 0.95
    assert li.core.hnsw.efSearch == 16  # index untouched


def test_sweep_values_deduplicated_and_sorted(ivf_setup) -> None:
    li, _, _, qs, truth = ivf_setup
    res = W.sweep(li, qs, truth, values=[4, 1, 4], k=K)
    assert [p.value for p in res.points] == [1, 4]


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"values": []}, "at least one"),
        ({"values": [NLIST + 1]}, "nprobe must be"),
        ({"param": "efSearch"}, "HNSW"),
        ({"k": 0}, "k must be"),
        ({"k": 5}, "truth must have shape"),
    ],
)
def test_sweep_errors(ivf_setup, kwargs, match) -> None:
    li, _, _, qs, truth = ivf_setup
    args = {"k": K, **kwargs}
    with pytest.raises(ValueError, match=match):
        W.sweep(li, qs, truth, **args)


def test_pareto() -> None:
    pts = [
        W.SweepPoint(1, 0.5, 0.1, 0.2),
        W.SweepPoint(2, 0.7, 0.2, 0.3),
        W.SweepPoint(4, 0.6, 0.3, 0.4),  # dominated by value 2
        W.SweepPoint(8, 0.9, 0.5, 0.6),
        W.SweepPoint(16, 0.9, 0.9, 1.0),  # same recall, slower: dominated
    ]
    res = W.SweepResult(W.SweepParam.NPROBE, 10, 5, "sampled", False, pts)
    assert [p.value for p in res.pareto()] == [1, 2, 8]


@pytest.mark.parametrize(
    ("spec", "param", "read"),
    [
        ("IDMap,IVF8,Flat", "nprobe", lambda i: faiss.extract_index_ivf(i).nprobe),
        ("PCA8,IVF8,Flat", "nprobe", lambda i: faiss.extract_index_ivf(i).nprobe),
        ("IVF8,PQ4x4,RFlat", "nprobe", lambda i: faiss.extract_index_ivf(i).nprobe),
        (
            "IDMap,HNSW16",
            "efSearch",
            lambda i: faiss.downcast_index(faiss.downcast_index(i).index).hnsw.efSearch,
        ),
    ],
)
def test_tuner_snippet_parameter_space_works_through_wrappers(spec, param, read) -> None:
    # The Tuner's "Apply it" snippet tells users to use ParameterSpace; keep that true.
    x = np.random.default_rng(0).standard_normal((1000, 16)).astype(np.float32)
    index = faiss.index_factory(16, spec)
    index.train(x)
    if spec.startswith("IDMap"):
        index.add_with_ids(x, np.arange(1000))
    else:
        index.add(x)
    faiss.ParameterSpace().set_index_parameter(index, param, 7)
    assert read(index) == 7


def test_repeated_timings_and_provenance(ivf_setup, monkeypatch) -> None:
    li, _, _, qs, truth = ivf_setup
    original = li.index.search
    calls = []

    def record(q, k, **kwargs):
        calls.append(len(q))
        return original(q, k, **kwargs)

    monkeypatch.setattr(li.index, "search", record)
    result = W.sweep(li, qs, truth, values=[1, 2], k=K, repeats=4, seed=17)
    assert calls.count(1) == 2 * len(qs) * 4
    assert calls.count(len(qs)) == 2  # each setting gets a warmup
    assert result.repeats == 4
    assert result.seed == 17
    assert result.query_seed == 0
    assert result.query_sha256 == W.query_fingerprint(qs)
    assert result.environment["threads"] == 1
    assert result.environment["faiss"] == faiss.__version__


def test_seed_changes_queries_and_fingerprint(ivf_setup) -> None:
    _, source, _, _, _ = ivf_setup
    a, b = W.sample_queries(source, 20, seed=1), W.sample_queries(source, 20, seed=2)
    assert W.query_fingerprint(a) != W.query_fingerprint(b)
    assert W.query_fingerprint(a) == W.query_fingerprint(W.sample_queries(source, 20, seed=1))
    assert W.query_fingerprint(a) != W.query_fingerprint(W.QuerySet(a.vectors, None, "given"))


@pytest.mark.parametrize("kwargs", [{"repeats": 0}, {"seed": -1}])
def test_invalid_measurement_settings(ivf_setup, kwargs) -> None:
    li, _, _, qs, truth = ivf_setup
    with pytest.raises(ValueError, match="repeats"):
        W.sweep(li, qs, truth, **kwargs)
