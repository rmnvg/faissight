import gc
import sys

import faiss
import numpy as np
import pytest

from faissight.core._faiss import FaissNotInstalledError, import_faiss
from faissight.core.loader import IndexLoadError, load_index
from faissight.core.types import IndexKind, Metric
from tests.conftest import SMALL

N, D, NLIST = SMALL["n"], SMALL["d"], SMALL["nlist"]


@pytest.mark.parametrize(
    ("name", "kind", "chain", "metric"),
    [
        ("flat_l2", IndexKind.FLAT, ["IndexFlatL2"], Metric.L2),
        ("ivf_flat", IndexKind.IVF_FLAT, ["IndexIVFFlat"], Metric.L2),
        ("ivf_pq", IndexKind.IVF_PQ, ["IndexIVFPQ"], Metric.L2),
        ("ivf_sq8", IndexKind.IVF_SQ, ["IndexIVFScalarQuantizer"], Metric.L2),
        ("hnsw_flat", IndexKind.HNSW_FLAT, ["IndexHNSWFlat"], Metric.L2),
        ("idmap_flat", IndexKind.FLAT, ["IndexIDMap", "IndexFlatL2"], Metric.L2),
        ("pca_ivf_flat", IndexKind.IVF_FLAT, ["IndexPreTransform", "IndexIVFFlat"], Metric.L2),
        ("ivf_flat_ip", IndexKind.IVF_FLAT, ["IndexIVFFlat"], Metric.IP),
    ],
)
def test_load_synthetic(synthetic, name, kind, chain, metric) -> None:
    li = load_index(synthetic[name])
    assert li.kind is kind
    assert li.class_chain == chain
    assert li.metric is metric
    assert li.d == D
    assert li.ntotal == N
    assert li.is_trained
    assert li.is_supported
    assert li.unsupported_reason is None
    assert li.path == synthetic[name]
    assert (li.ivf is not None) == kind.is_ivf


def test_ivf_params(synthetic) -> None:
    p = load_index(synthetic["ivf_flat"]).params
    assert p.nlist == NLIST
    assert p.nprobe == 1
    assert p.pq_m is None
    assert p.hnsw_m is None


def test_ivf_pq_params(synthetic) -> None:
    p = load_index(synthetic["ivf_pq"]).params
    assert (p.nlist, p.pq_m, p.pq_nbits, p.by_residual) == (NLIST, 8, 8, True)


def test_ivf_sq_params(synthetic) -> None:
    assert load_index(synthetic["ivf_sq8"]).params.sq_type == "8bit"


def test_hnsw_params(synthetic) -> None:
    li = load_index(synthetic["hnsw_flat"])
    p = li.params
    assert p.hnsw_m == SMALL["hnsw_m"]
    assert p.ef_search == 16
    assert p.ef_construction == 40
    assert p.max_level is not None
    assert p.max_level >= 1
    assert p.entry_point is not None
    assert 0 <= p.entry_point < N
    assert set(p.as_dict()) == {
        "hnsw_m",
        "ef_search",
        "ef_construction",
        "max_level",
        "entry_point",
    }


def test_idmap_ids(synthetic) -> None:
    li = load_index(synthetic["idmap_flat"])
    assert li.has_id_map
    np.testing.assert_array_equal(li.ids, np.load(synthetic["ids_idmap"]))
    np.testing.assert_array_equal(li.user_ids(np.array([0, 2, -1])), [li.ids[0], li.ids[2], -1])


def test_user_ids_without_id_map(synthetic) -> None:
    li = load_index(synthetic["flat_l2"])
    assert not li.has_id_map
    np.testing.assert_array_equal(li.user_ids(np.array([3, -1])), [3, -1])


def test_pretransform(synthetic) -> None:
    li = load_index(synthetic["pca_ivf_flat"])
    assert [(t.name, t.d_in, t.d_out) for t in li.transforms] == [("PCAMatrix", D, D // 2)]
    assert li.core_d == D // 2
    assert li.ivf.quantizer.d == D // 2


def test_idmap2_over_ivf_in_memory() -> None:
    x = np.random.default_rng(0).standard_normal((500, 8)).astype(np.float32)
    ivf = faiss.IndexIVFFlat(faiss.IndexFlatL2(8), 8, 4)
    ivf.train(x)
    index = faiss.IndexIDMap2(ivf)
    index.add_with_ids(x, np.arange(500, dtype=np.int64) + 10)
    li = load_index(index)
    assert li.path is None
    assert li.class_chain == ["IndexIDMap2", "IndexIVFFlat"]
    assert li.kind is IndexKind.IVF_FLAT
    assert li.ids[0] == 10


def test_refine() -> None:
    x = np.random.default_rng(0).standard_normal((500, 8)).astype(np.float32)
    index = faiss.index_factory(8, "IVF4,PQ4x4,RFlat")
    index.train(x)
    index.add(x)
    li = load_index(index)
    assert li.has_refine
    assert li.class_chain == ["IndexRefineFlat", "IndexIVFPQ"]
    assert li.kind is IndexKind.IVF_PQ


def test_hnsw_other() -> None:
    li = load_index(faiss.IndexHNSWSQ(8, faiss.ScalarQuantizer.QT_8bit, 16))
    assert li.kind is IndexKind.HNSW_OTHER


def test_core_outlives_temporaries() -> None:
    # The wrapper chain is built from temporaries; LoadedIndex must keep the root alive.
    li = load_index(faiss.index_factory(16, "IDMap,Flat"))
    gc.collect()
    li.core.add(np.zeros((2, 16), dtype=np.float32))
    assert li.core.ntotal == 2


@pytest.mark.parametrize(
    "factory",
    [
        lambda: faiss.IndexLSH(16, 32),
        lambda: faiss.IndexPQ(16, 4, 8),
        lambda: faiss.IndexIVFPQFastScan(faiss.IndexFlatL2(16), 16, 4, 4, 4),
        lambda: faiss.IndexFlat(16, faiss.METRIC_L1),
    ],
)
def test_unsupported_types(factory) -> None:
    li = load_index(factory())
    assert li.kind is IndexKind.UNSUPPORTED
    assert not li.is_supported
    assert li.unsupported_reason


def test_binary_index_file(binary_index_path) -> None:
    li = load_index(binary_index_path)
    assert li.kind is IndexKind.UNSUPPORTED
    assert "Binary" in li.unsupported_reason
    assert li.d == 64


def test_binary_index_in_memory() -> None:
    assert load_index(faiss.IndexBinaryFlat(64)).kind is IndexKind.UNSUPPORTED


def test_missing_file(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        load_index(tmp_path / "nope.index")


def test_garbage_file(tmp_path) -> None:
    path = tmp_path / "garbage.index"
    path.write_bytes(b"not a faiss index at all")
    with pytest.raises(IndexLoadError):
        load_index(path)


def test_wrong_type() -> None:
    with pytest.raises(TypeError):
        load_index(np.zeros((3, 3)))


def test_faiss_missing(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "faiss", None)
    with pytest.raises(FaissNotInstalledError, match="faissight\\[faiss-cpu\\]"):
        import_faiss()
