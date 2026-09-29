import faiss
import numpy as np
import pytest

from faissight.core import vectors as V
from faissight.core.loader import load_index
from tests.conftest import SMALL

N, D, NLIST = SMALL["n"], SMALL["d"], SMALL["nlist"]


@pytest.fixture(scope="module")
def x(synthetic):
    return np.load(synthetic["vectors"])


def test_from_arrays_default_ids(synthetic, x) -> None:
    src = V.from_arrays(load_index(synthetic["flat_l2"]), x)
    assert not src.reconstructed
    assert len(src) == N
    np.testing.assert_array_equal(src.ids, np.arange(N))
    np.testing.assert_array_equal(src.get([5, 0]), x[[5, 0]])


def test_from_arrays_sorts_by_id(synthetic, x) -> None:
    li = load_index(synthetic["flat_l2"])
    ids = np.arange(N, dtype=np.int64)[::-1] * 2
    src = V.from_arrays(li, x, ids)
    assert (np.diff(src.ids) > 0).all()
    np.testing.assert_array_equal(src.get([ids[3]]), x[[3]])
    np.testing.assert_array_equal(src.positions([ids[3], 1, -7]) >= 0, [True, False, False])
    with pytest.raises(KeyError):
        src.get([1])


@pytest.mark.parametrize(
    ("make", "match"),
    [
        (lambda x: x[:, :-1], "dimension"),
        (lambda x: x[:-1], "vectors but the index holds"),
        (lambda x: x[0], "2-D"),
    ],
)
def test_from_arrays_errors(synthetic, x, make, match) -> None:
    with pytest.raises(V.VectorMismatchError, match=match) as e:
        V.from_arrays(load_index(synthetic["flat_l2"]), make(x))
    assert e.value.hint


def test_from_arrays_bad_ids(synthetic, x) -> None:
    li = load_index(synthetic["flat_l2"])
    with pytest.raises(V.VectorMismatchError, match="ids for"):
        V.from_arrays(li, x, np.arange(N - 1))
    with pytest.raises(V.VectorMismatchError, match="duplicates"):
        V.from_arrays(li, x, np.zeros(N, dtype=np.int64))


@pytest.mark.parametrize("name", ["flat_l2", "ivf_flat", "hnsw_flat"])
def test_reconstruct_exact_for_flat_storage(synthetic, x, name) -> None:
    src = V.reconstruct_all(load_index(synthetic[name]))
    assert src.reconstructed
    np.testing.assert_array_equal(src.ids, np.arange(N))
    np.testing.assert_allclose(src.vectors, x, atol=1e-5)


def test_reconstruct_idmap(synthetic, x) -> None:
    src = V.reconstruct_all(load_index(synthetic["idmap_flat"]))
    ids = np.load(synthetic["ids_idmap"])
    np.testing.assert_array_equal(src.ids, np.sort(ids))
    np.testing.assert_allclose(src.get(ids[:10]), x[:10], atol=1e-5)


def test_reconstruct_lossy(synthetic, x) -> None:
    pq = V.reconstruct_all(load_index(synthetic["ivf_pq"]))
    sq = V.reconstruct_all(load_index(synthetic["ivf_sq8"]))
    err_pq = np.square(pq.vectors - x).sum(1).mean()
    err_sq = np.square(sq.vectors - x).sum(1).mean()
    assert 0 < err_sq < err_pq


def test_reconstruct_pretransform_input_space(synthetic, x) -> None:
    src = V.reconstruct_all(load_index(synthetic["pca_ivf_flat"]))
    assert src.vectors.shape == (N, D)
    # PCA to D/2 dims loses information but keeps most of the blob structure.
    rel = np.square(src.vectors - x).sum() / np.square(x - x.mean(0)).sum()
    assert rel < 0.5


def test_reconstruct_sparse_ids_uses_hashtable() -> None:
    xs = np.random.default_rng(0).standard_normal((500, 8)).astype(np.float32)
    index = faiss.IndexIVFFlat(faiss.IndexFlatL2(8), 8, 4)
    index.train(xs)
    index.add_with_ids(xs, np.arange(500, dtype=np.int64) * 5 + 3)
    src = V.reconstruct_all(load_index(index))
    assert index.direct_map.type == faiss.DirectMap.Hashtable
    np.testing.assert_allclose(src.get([3, 8]), xs[:2])


def test_direct_map_does_not_change_search(synthetic, x) -> None:
    li = load_index(synthetic["ivf_flat"])
    before = li.index.search(x[:20], 10)
    V.reconstruct_all(li)
    after = li.index.search(x[:20], 10)
    np.testing.assert_array_equal(before[1], after[1])


def test_reconstruct_unsupported() -> None:
    with pytest.raises(ValueError, match="Cannot reconstruct"):
        V.reconstruct_all(load_index(faiss.IndexLSH(8, 16)))


def test_from_arrays_does_not_copy_sorted_input(synthetic, x) -> None:
    # Large vector sets must not be duplicated when ids are already in order.
    li = load_index(synthetic["flat_l2"])
    assert np.shares_memory(V.from_arrays(li, x).vectors, x)
    shuffled = V.from_arrays(li, x, np.arange(N)[::-1].copy())
    assert not np.shares_memory(shuffled.vectors, x)
    np.testing.assert_array_equal(shuffled.get([0]), x[[N - 1]])  # id 0 is the last row
