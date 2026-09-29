import faiss
import numpy as np
import pytest

from faissight.core import ivf
from faissight.core import pq as P
from faissight.core import vectors as V
from faissight.core.loader import load_index
from faissight.core.types import Metric
from tests.conftest import SMALL

N, D, NLIST = SMALL["n"], SMALL["d"], SMALL["nlist"]


@pytest.fixture(scope="module")
def x(synthetic):
    return np.load(synthetic["vectors"])


def _report(synthetic, x, name):
    li = load_index(synthetic[name])
    a = ivf.assignments(li) if li.kind.is_ivf else None
    return li, P.analyze(li, V.from_arrays(li, x), V.reconstruct_all(li), a)


def test_error_ordering_flat_sq_pq(synthetic, x) -> None:
    # Acceptance: ~0 for IVFFlat, clearly > 0 for IVFPQ, and SQ8 below PQ(m=8).
    _, flat = _report(synthetic, x, "ivf_flat")
    _, sq = _report(synthetic, x, "ivf_sq8")
    _, pq = _report(synthetic, x, "ivf_pq")
    assert flat.mean == pytest.approx(0.0, abs=1e-6)
    assert pq.mean > 1.0
    assert 0 < sq.mean < pq.mean
    assert flat.distortion.near_correlation == pytest.approx(1.0)
    assert pq.distortion.near_correlation < sq.distortion.near_correlation


@pytest.mark.parametrize("name", ["flat_l2", "hnsw_flat", "idmap_flat"])
def test_exact_storage_has_no_error(synthetic, x, name) -> None:
    li = load_index(synthetic[name])
    ids = np.load(synthetic["ids_idmap"]) if name == "idmap_flat" else None
    rep = P.analyze(li, V.from_arrays(li, x, ids), V.reconstruct_all(li))
    assert rep.mean == pytest.approx(0.0, abs=1e-6)
    assert rep.per_list() == []


def test_pretransform_error_includes_transform_loss(synthetic, x) -> None:
    _, rep = _report(synthetic, x, "pca_ivf_flat")
    assert rep.mean > 0  # flat codes, but PCA to D/2 loses information


def test_summaries(synthetic, x) -> None:
    li, rep = _report(synthetic, x, "ivf_pq")
    assert rep.errors.shape == rep.relative.shape == (N,)
    assert (rep.relative >= 0).all()
    q = rep.quantiles()
    assert q["median"] <= q["p95"] <= q["max"] == pytest.approx(rep.errors.max())
    edges, counts = rep.histogram(bins=20)
    assert len(edges) == 21
    assert counts.sum() == N
    per_list = rep.per_list()
    assert sum(size for _, size, _ in per_list) == N
    sizes = ivf.list_sizes(li)
    for list_no, size, mean in per_list:
        assert size == sizes[list_no]
        assert mean == pytest.approx(rep.errors[rep.list_nos == list_no].mean())
    worst = rep.worst(50)
    assert len(worst) == 50
    errs = [e for _, e, _ in worst]
    assert errs == sorted(errs, reverse=True)
    assert errs[0] == pytest.approx(rep.errors.max())
    assert rep.code_size == 8  # PQ m=8, nbits=8
    assert rep.raw_bytes == 4 * D


def test_code_sizes(synthetic) -> None:
    assert P.code_size(load_index(synthetic["ivf_sq8"])) == D
    assert P.code_size(load_index(synthetic["ivf_flat"])) == 4 * D
    assert P.code_size(load_index(synthetic["hnsw_flat"])) == 4 * D


def test_distortion_pairs(synthetic, x) -> None:
    li = load_index(synthetic["ivf_flat"])
    d = P.distance_distortion(
        V.from_arrays(li, x), V.reconstruct_all(li), li.metric, n_queries=10, n_near=5, n_random=5
    )
    assert len(d.true) == len(d.approx) == len(d.near) == 100
    assert d.near.sum() == 50
    np.testing.assert_allclose(d.true, d.approx, rtol=1e-4, atol=1e-3)
    # Near pairs are closer than random ones.
    assert d.true[d.near].mean() < d.true[~d.near].mean()


def test_distortion_inner_product() -> None:
    rng = np.random.default_rng(0)
    xs = rng.standard_normal((1000, 16)).astype(np.float32)
    xs /= np.linalg.norm(xs, axis=1, keepdims=True)
    index = faiss.IndexIVFPQ(faiss.IndexFlatIP(16), 16, 8, 4, 8, faiss.METRIC_INNER_PRODUCT)
    index.train(xs)
    index.add(xs)
    li = load_index(index)
    rep = P.analyze(li, V.from_arrays(li, xs), V.reconstruct_all(li))
    assert li.metric is Metric.IP
    d = rep.distortion
    assert d.true.max() <= 1.0 + 1e-5  # inner products of unit vectors
    assert d.true[d.near].mean() > d.true[~d.near].mean()  # higher = closer for IP


def test_requires_raw_vectors(synthetic) -> None:
    li = load_index(synthetic["ivf_pq"])
    stored = V.reconstruct_all(li)
    with pytest.raises(P.RawVectorsRequiredError, match="--vectors"):
        P.analyze(li, None, stored)
    with pytest.raises(P.RawVectorsRequiredError):
        P.analyze(li, stored, stored)  # reconstructed isn't raw


def test_mismatched_ids(synthetic, x) -> None:
    li = load_index(synthetic["ivf_flat"])
    raw = V.from_arrays(li, x, np.arange(N) + 1)
    with pytest.raises(ValueError, match="same ids"):
        P.reconstruction_errors(raw, V.reconstruct_all(li))
