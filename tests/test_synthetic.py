import faiss
import numpy as np
import pytest

from tests.conftest import SMALL


def test_all_indexes_written(synthetic, make_synthetic) -> None:
    for name in make_synthetic.INDEX_NAMES:
        index = faiss.read_index(str(synthetic[name]))
        assert index.ntotal == SMALL["n"], name
        assert index.d == SMALL["d"], name


def test_side_files(synthetic) -> None:
    x = np.load(synthetic["vectors"])
    assert x.shape == (SMALL["n"], SMALL["d"])
    assert x.dtype == np.float32
    assert np.load(synthetic["queries"]).shape == (SMALL["n_queries"], SMALL["d"])
    np.testing.assert_allclose(
        np.linalg.norm(np.load(synthetic["vectors_normalized"]), axis=1), 1.0, rtol=1e-5
    )
    ids = np.load(synthetic["ids_idmap"])
    assert len(np.unique(ids)) == SMALL["n"]
    assert synthetic["chunks"].read_text().count("\n") == SMALL["n"]


@pytest.mark.parametrize("seed", [0, 1])
def test_blobs_deterministic(make_synthetic, seed) -> None:
    a, _ = make_synthetic.make_blobs(100, 8, seed=seed)
    b, _ = make_synthetic.make_blobs(100, 8, seed=seed)
    np.testing.assert_array_equal(a, b)
