import sys
import threading
import types
from concurrent.futures import ThreadPoolExecutor

import faiss
import numpy as np
import pytest

from faissight import session as session_mod
from faissight.core import vectors as V
from faissight.core.projection import ProjectionUnavailableError
from faissight.session import InputError, Session
from tests.conftest import SMALL

N, D, NLIST = SMALL["n"], SMALL["d"], SMALL["nlist"]


@pytest.fixture
def full(synthetic, tmp_path):
    return Session(
        synthetic["ivf_flat"],
        vectors=synthetic["vectors"],
        metadata=synthetic["chunks"],
        queries=synthetic["queries"],
        cache_root=tmp_path,
    )


def test_inputs_loaded(full) -> None:
    assert full.has_raw_vectors
    assert not full.source.reconstructed
    assert full.queries.shape == (SMALL["n_queries"], D)
    assert len(full.metadata) == N
    assert full.metadata_coverage() == 1.0
    assert full.assignments is not None
    assert len(full.index_sha1) == 40


def test_query_by_id_with_compare(full) -> None:
    rep = full.query(id=5, k=10, nprobe=NLIST)
    assert rep.recall == 1.0
    assert 5 not in rep.result.ids
    assert rep.ivf_trace is not None
    assert not rep.truth_reconstructed


def test_query_without_compare(full) -> None:
    rep = full.query(vector=np.zeros(D), k=3, compare=False)
    assert rep.truth is None
    assert rep.ivf_trace is None


def test_query_text_with_callable(synthetic) -> None:
    s = Session(synthetic["flat_l2"], embedder=lambda text: np.ones(D))
    assert s.embedder_name == "<lambda>"
    rep = s.query(text="anything", k=5)
    assert rep.query.text == "anything"
    assert rep.truth_reconstructed  # no raw vectors: ground truth on reconstructed


def test_query_text_with_model_name(synthetic, monkeypatch) -> None:
    seen = {}

    class SentenceTransformer:
        def __init__(self, name):
            seen["name"] = name

        def encode(self, texts, normalize_embeddings, convert_to_numpy):
            seen["normalize"] = normalize_embeddings
            return np.ones((1, D), dtype=np.float32)

    module = types.ModuleType("sentence_transformers")
    module.SentenceTransformer = SentenceTransformer
    monkeypatch.setitem(sys.modules, "sentence_transformers", module)

    s = Session(
        synthetic["ivf_flat_ip"], vectors=synthetic["vectors_normalized"], embedder="my-model"
    )
    assert s.embedder_name == "my-model"
    s.start_background()
    rep = s.query(text="hello", k=5)
    assert seen == {"name": "my-model", "normalize": True}  # stored vectors are unit-norm
    assert rep.result.ids.shape == (5,)


def test_normalize_text_rules(synthetic) -> None:
    assert not Session(synthetic["flat_l2"], vectors=synthetic["vectors"])._should_normalize_text()
    assert Session(synthetic["ivf_flat_ip"])._should_normalize_text()  # IP, no vectors
    assert not Session(synthetic["flat_l2"])._should_normalize_text()
    assert not Session(synthetic["ivf_flat_ip"], normalize_text=False)._should_normalize_text()


def test_missing_embedder_package_is_deferred(synthetic, monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)
    s = Session(synthetic["flat_l2"], embedder="m")
    s.start_background()  # must not raise
    with pytest.raises(Exception, match="sentence-transformers"):
        s.query(text="x")


def test_reconstructed_source_computed_once(synthetic, monkeypatch) -> None:
    calls = []
    real = V.reconstruct_all

    def counting(li):
        calls.append(1)
        return real(li)

    monkeypatch.setattr(session_mod, "reconstruct_all", counting)
    s = Session(synthetic["ivf_flat"])
    queries = np.load(synthetic["queries"])
    with ThreadPoolExecutor(8) as pool:
        reports = list(pool.map(lambda i: s.query(id=int(i), k=5, nprobe=4), range(16)))
    assert calls == [1]
    assert all(r.truth_reconstructed for r in reports)
    assert s.query(vector=queries[0], k=5, nprobe=NLIST).recall == 1.0


def test_known_ids(synthetic) -> None:
    ids = np.load(synthetic["ids_idmap"])
    np.testing.assert_array_equal(Session(synthetic["idmap_flat"]).known_ids(), ids)
    np.testing.assert_array_equal(Session(synthetic["ivf_flat"]).known_ids(), np.arange(N))
    np.testing.assert_array_equal(Session(synthetic["flat_l2"]).known_ids(), np.arange(N))
    assert Session(synthetic["flat_l2"]).metadata_coverage() is None


def test_idmap_metadata_coverage(synthetic) -> None:
    # chunks.jsonl is keyed 0..n-1 but idmap_flat uses sparse ids: partial coverage.
    s = Session(synthetic["idmap_flat"], metadata=synthetic["chunks"])
    assert 0 < s.metadata_coverage() < 1


def test_projection_job_and_place(full) -> None:
    job = full.projection_job("pca", 2)
    assert job.wait(10)
    proj = job.result
    assert full.projection_job("pca", 2) is job
    xy = full.place(proj, full.source.get([proj.ids[0]])[0])
    np.testing.assert_allclose(xy, proj.coords[0], atol=1e-4)


def test_projection_disk_cache_used(synthetic, tmp_path) -> None:
    kwargs = {"vectors": synthetic["vectors"], "cache_root": tmp_path}
    Session(synthetic["ivf_flat"], **kwargs).projection_job().wait(10)
    assert len(list(tmp_path.rglob("*.npz"))) == 1
    Session(
        synthetic["ivf_flat"],
        disk_cache=False,
        vectors=synthetic["vectors"],
        cache_root=tmp_path / "x",
    ).projection_job().wait(10)
    assert not (tmp_path / "x").exists()


def test_start_background_starts_pca(full) -> None:
    full.start_background()
    job = full.jobs.get(("projection", "pca", 2))
    assert job is not None
    assert job.wait(10)


def test_umap_unavailable(full, monkeypatch) -> None:
    monkeypatch.setattr(session_mod.importlib.util, "find_spec", lambda name: None)
    with pytest.raises(ProjectionUnavailableError):
        full.projection_job("umap", 2)


def test_projection_bad_dims(full) -> None:
    with pytest.raises(ValueError, match="dims"):
        full.projection_job("pca", 4)


def test_unsupported_index_session(binary_index_path) -> None:
    s = Session(binary_index_path)
    assert not s.li.is_supported
    s.start_background()  # no-op, no crash
    with pytest.raises(ValueError, match="Cannot project"):
        s.projection_job()


def test_in_memory_inputs() -> None:
    x = np.random.default_rng(0).standard_normal((300, 8)).astype(np.float32)
    index = faiss.IndexFlatL2(8)
    index.add(x)
    s = Session(index, vectors=x, metadata=[{"id": 1, "text": "hi"}], queries=x[:3])
    assert s.metadata.get(1) == {"text": "hi"}
    assert s.query(id=1, k=3).recall == 1.0


@pytest.mark.parametrize(
    ("kwargs", "code"),
    [
        ({"vectors": np.zeros((N, D + 1), dtype=np.float32)}, "VECTOR_MISMATCH"),
        ({"vectors": np.zeros((N - 1, D), dtype=np.float32)}, "VECTOR_MISMATCH"),
        ({"ids": np.arange(N)}, "IDS_WITHOUT_VECTORS"),
        ({"queries": np.zeros((5, D + 2))}, "QUERY_MISMATCH"),
        ({"vectors": "/nope/v.npy"}, "FILE_NOT_FOUND"),
        ({"metadata": [{"no_id": 1}]}, "BAD_METADATA"),
        ({"max_points": 0}, "BAD_MAX_POINTS"),
    ],
)
def test_input_validation(synthetic, kwargs, code) -> None:
    with pytest.raises(InputError) as e:
        Session(synthetic["ivf_flat"], **kwargs)
    assert e.value.code == code
    assert e.value.hint


def test_bad_npy_file(synthetic, tmp_path) -> None:
    p = tmp_path / "v.npy"
    p.write_bytes(b"not numpy")
    with pytest.raises(InputError) as e:
        Session(synthetic["ivf_flat"], vectors=p)
    assert e.value.code == "BAD_NPY"


def test_concurrent_projection_and_queries(synthetic) -> None:
    # Reconstruction (direct map) and projection run while queries hit the index.
    s = Session(synthetic["ivf_pq"])
    errors = []

    def hammer():
        try:
            for i in range(20):
                s.query(id=i, k=5, nprobe=4)
        except Exception as e:  # pragma: no cover - reported below
            errors.append(e)

    threads = [threading.Thread(target=hammer) for _ in range(4)]
    s.start_background()
    for t in threads:
        t.start()
    for t in threads:
        t.join(30)
    assert not errors
    assert s.jobs.get(("projection", "pca", 2)).wait(10)
