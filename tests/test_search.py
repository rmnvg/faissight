import faiss
import numpy as np
import pytest

from faissight.core import search as S
from faissight.core.loader import load_index
from faissight.core.types import Metric
from tests.conftest import SMALL

N, D, NLIST = SMALL["n"], SMALL["d"], SMALL["nlist"]


@pytest.fixture(scope="module")
def data(synthetic):
    return {
        "x": np.load(synthetic["vectors"]),
        "xn": np.load(synthetic["vectors_normalized"]),
        "queries": np.load(synthetic["queries"]),
        "ids_idmap": np.load(synthetic["ids_idmap"]),
    }


# --- search -----------------------------------------------------------------


def test_search_ivf_uses_params_without_mutating(synthetic, data) -> None:
    li = load_index(synthetic["ivf_flat"])
    r = S.search(li, data["queries"][0], 10, nprobe=NLIST)
    assert li.ivf.nprobe == 1
    assert r.params == {"nprobe": NLIST}
    assert r.ids.shape == (10,)
    assert r.metric is Metric.L2
    assert r.latency_ms >= 0
    # Same as FAISS with the index attribute set.
    li.ivf.nprobe = NLIST
    _, expected = li.index.search(data["queries"][:1], 10)
    np.testing.assert_array_equal(r.ids, expected[0])


def test_search_defaults_to_index_params(synthetic, data) -> None:
    li = load_index(synthetic["ivf_flat"])
    assert S.search(li, data["queries"][0], 5).params == {"nprobe": 1}
    h = load_index(synthetic["hnsw_flat"])
    assert S.search(h, data["queries"][0], 5).params == {"efSearch": 16}
    f = load_index(synthetic["flat_l2"])
    assert S.search(f, data["queries"][0], 5).params == {}


def test_search_hnsw_ef(synthetic, data) -> None:
    li = load_index(synthetic["hnsw_flat"])
    r = S.search(li, data["queries"][0], 10, ef_search=128)
    assert r.params == {"efSearch": 128}
    assert li.core.hnsw.efSearch == 16


def test_search_idmap_returns_user_ids(synthetic, data) -> None:
    li = load_index(synthetic["idmap_flat"])
    r = S.search(li, data["x"][3], 1)
    assert r.ids[0] == data["ids_idmap"][3]


def test_search_pretransform_nprobe(synthetic, data) -> None:
    # nprobe must reach the IVF through the PreTransform wrapper. Blob queries sit inside
    # one cell here, so use random off-blob queries that straddle cell boundaries.
    li = load_index(synthetic["pca_ivf_flat"])
    queries = np.random.default_rng(5).normal(scale=4.0, size=(10, D)).astype(np.float32)
    changed = [
        not np.array_equal(S.search(li, q, 10, nprobe=1).ids, S.search(li, q, 10, nprobe=NLIST).ids)
        for q in queries
    ]
    assert any(changed)


def test_search_refine_keeps_k_factor() -> None:
    x = np.random.default_rng(0).standard_normal((1000, 16)).astype(np.float32)
    index = faiss.index_factory(16, "IVF8,PQ4x4,RFlat")
    index.train(x)
    index.add(x)
    faiss.downcast_index(index).k_factor = 4
    li = load_index(index)
    assert li.refine_k_factor == 4
    r = S.search(li, x[0], 5, nprobe=8)
    faiss.extract_index_ivf(index).nprobe = 8
    _, expected = index.search(x[:1], 5)
    np.testing.assert_array_equal(r.ids, expected[0])


def test_search_exclude_self(synthetic, data) -> None:
    li = load_index(synthetic["flat_l2"])
    with_self = S.search(li, data["x"][7], 5)
    assert with_self.ids[0] == 7
    r = S.search(li, data["x"][7], 5, exclude_id=7)
    assert 7 not in r.ids
    assert len(r.ids) == 5
    np.testing.assert_array_equal(r.ids[:4], with_self.ids[1:])


def test_search_ip_metric(synthetic, data) -> None:
    li = load_index(synthetic["ivf_flat_ip"])
    r = S.search(li, data["xn"][0], 5, nprobe=NLIST)
    assert r.metric is Metric.IP
    assert (np.diff(r.distances) <= 0).all()  # similarities, best first


@pytest.mark.parametrize(
    ("name", "kwargs", "match"),
    [
        ("ivf_flat", {"nprobe": 0}, "nprobe must be"),
        ("ivf_flat", {"nprobe": NLIST + 1}, "nprobe must be"),
        ("ivf_flat", {"ef_search": 32}, "efSearch only"),
        ("hnsw_flat", {"nprobe": 4}, "nprobe only"),
        ("hnsw_flat", {"ef_search": 0}, "efSearch must be"),
        ("flat_l2", {"nprobe": 4}, "no search parameters"),
    ],
)
def test_search_param_errors(synthetic, data, name, kwargs, match) -> None:
    with pytest.raises(ValueError, match=match):
        S.search(load_index(synthetic[name]), data["queries"][0], 5, **kwargs)


def test_search_bad_inputs(synthetic) -> None:
    li = load_index(synthetic["flat_l2"])
    with pytest.raises(ValueError, match="dimension"):
        S.search(li, np.zeros(D + 1), 5)
    with pytest.raises(ValueError, match="k must be"):
        S.search(li, np.zeros(D), 0)
    with pytest.raises(ValueError, match="Cannot search"):
        S.search(load_index(faiss.IndexLSH(8, 16)), np.zeros(8), 5)
