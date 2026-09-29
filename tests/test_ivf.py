import faiss
import numpy as np
import pytest

from faissight.core import ivf
from faissight.core.loader import load_index
from tests.conftest import SMALL

N, D, NLIST = SMALL["n"], SMALL["d"], SMALL["nlist"]
IVF_NAMES = ["ivf_flat", "ivf_pq", "ivf_sq8", "pca_ivf_flat", "ivf_flat_ip"]


@pytest.fixture(scope="module", params=IVF_NAMES)
def loaded(request, synthetic):
    return load_index(synthetic[request.param])


def test_sizes_sum_to_ntotal(loaded) -> None:
    sizes = ivf.list_sizes(loaded)
    assert sizes.shape == (NLIST,)
    assert sizes.sum() == loaded.ntotal


def test_every_id_in_exactly_one_list(loaded) -> None:
    members = np.concatenate([ivf.list_members(loaded, i) for i in range(NLIST)])
    assert len(members) == loaded.ntotal
    np.testing.assert_array_equal(np.sort(members), np.arange(N))


def test_members_match_sizes(loaded) -> None:
    sizes = ivf.list_sizes(loaded)
    for i in range(NLIST):
        assert len(ivf.list_members(loaded, i)) == sizes[i]


def test_centroids_shape(loaded) -> None:
    c = ivf.centroids(loaded)
    assert c.shape == (NLIST, loaded.core_d)
    assert c.dtype == np.float32


def test_pretransform_centroids_in_transformed_space(synthetic) -> None:
    li = load_index(synthetic["pca_ivf_flat"])
    assert ivf.centroids(li).shape == (NLIST, D // 2)


def test_assignments_agree_with_members(loaded) -> None:
    a = ivf.assignments(loaded)
    assert len(a.ids) == loaded.ntotal
    for list_no in (0, NLIST // 2, NLIST - 1):
        members = ivf.list_members(loaded, list_no)
        assert (a.lookup(members) == list_no).all()


def test_assignments_match_quantizer(synthetic) -> None:
    # Stored vectors were assigned to their nearest centroid at add time.
    li = load_index(synthetic["ivf_flat"])
    x = np.load(synthetic["vectors"])
    _, nearest = li.ivf.quantizer.search(x, 1)
    np.testing.assert_array_equal(ivf.assignments(li).lookup(np.arange(N)), nearest[:, 0])


def test_lookup_unknown_ids(synthetic) -> None:
    a = ivf.assignments(load_index(synthetic["ivf_flat"]))
    np.testing.assert_array_equal(a.lookup([-5, N, N + 100]), [-1, -1, -1])


def test_imbalance_matches_faiss(loaded) -> None:
    ours = ivf.imbalance_factor(ivf.list_sizes(loaded))
    assert ours == pytest.approx(loaded.ivf.invlists.imbalance_factor())
    assert ours >= 1.0


def test_imbalance_edge_cases() -> None:
    assert ivf.imbalance_factor([5, 5, 5, 5]) == pytest.approx(1.0)
    assert ivf.imbalance_factor([20, 0, 0, 0]) == pytest.approx(4.0)
    assert ivf.imbalance_factor([0, 0]) == 0.0


def test_list_stats(synthetic) -> None:
    li = load_index(synthetic["ivf_flat"])
    stats = ivf.list_stats(li, top=5)
    sizes = stats.sizes
    assert stats.n_empty == int((sizes == 0).sum())
    assert (stats.min, stats.max) == (sizes.min(), sizes.max())
    assert stats.median == np.median(sizes)
    assert len(stats.top_lists) == 5
    assert stats.top_lists[0] == (int(sizes.argmax()), int(sizes.max()))
    assert [s for _, s in stats.top_lists] == sorted((s for _, s in stats.top_lists), reverse=True)
    # 5% of 16 lists rounds up to one list: the largest.
    assert stats.top_5pct_share == pytest.approx(sizes.max() / N)


def test_list_stats_empty_index() -> None:
    index = faiss.IndexIVFFlat(faiss.IndexFlatL2(8), 8, 4)
    index.train(np.random.default_rng(0).standard_normal((200, 8)).astype(np.float32))
    stats = ivf.list_stats(load_index(index))
    assert stats.n_empty == 4
    assert stats.imbalance_factor == 0.0
    assert stats.top_5pct_share == 0.0


def test_list_members_out_of_range(synthetic) -> None:
    li = load_index(synthetic["ivf_flat"])
    with pytest.raises(IndexError):
        ivf.list_members(li, NLIST)
    with pytest.raises(IndexError):
        ivf.list_members(li, -1)


@pytest.mark.parametrize("fn", [ivf.list_sizes, ivf.centroids, ivf.assignments])
def test_not_ivf(synthetic, fn) -> None:
    with pytest.raises(ivf.NotAnIVFIndexError):
        fn(load_index(synthetic["hnsw_flat"]))


def _ivf_with_ids(wrap: bool):
    x = np.random.default_rng(1).standard_normal((600, 8)).astype(np.float32)
    ids = np.arange(600, dtype=np.int64) * 13 + 500
    base = faiss.IndexIVFFlat(faiss.IndexFlatL2(8), 8, 8)
    base.train(x)
    index = faiss.IndexIDMap2(base) if wrap else base
    index.add_with_ids(x, ids)
    return index, x, ids


@pytest.mark.parametrize("wrap", [True, False], ids=["idmap2_over_ivf", "ivf_add_with_ids"])
def test_user_facing_ids(wrap) -> None:
    index, x, ids = _ivf_with_ids(wrap)
    li = load_index(index)
    assert li.has_id_map == wrap

    members = np.concatenate([ivf.list_members(li, i) for i in range(8)])
    np.testing.assert_array_equal(np.sort(members), ids)

    a = ivf.assignments(li)
    np.testing.assert_array_equal(a.ids, ids)
    _, nearest = li.ivf.quantizer.search(x, 1)
    np.testing.assert_array_equal(a.lookup(ids), nearest[:, 0])

    # Search results come back as user ids and must resolve to their list.
    _, found = index.search(x[:5], 1)
    assert (a.lookup(found[:, 0]) >= 0).all()
