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

    s = Session(synthetic["ivf_flat"])

    def counting(li):
        # Count only this session's calls: background jobs from earlier tests' sessions can
        # still be running and would hit the same module-level function.
        if li is s.li:
            calls.append(1)
        return real(li)

    monkeypatch.setattr(session_mod, "reconstruct_all", counting)
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


def test_sweep_queries_given_vs_sampled(synthetic) -> None:
    given = Session(synthetic["ivf_flat"], queries=synthetic["queries"]).sweep_queries(10)
    assert (given.origin, len(given)) == ("given", 10)
    sampled = Session(synthetic["ivf_flat"]).sweep_queries(25)
    assert (sampled.origin, len(sampled)) == ("sampled", 25)
    assert sampled.exclude_ids is not None


def test_sweep_job_reconstructed_truth(synthetic) -> None:
    s = Session(synthetic["ivf_pq"])
    job_id, job = s.sweep_job(values=[1, NLIST], n_queries=20)
    assert job.wait(30)
    assert job.result.truth_reconstructed
    assert s.get_sweep_job(job_id) is job
    assert s.get_sweep_job("missing") is None
    # Same parameters -> same id and job.
    assert s.sweep_job(values=[NLIST, 1], n_queries=20) == (job_id, job)


def test_sweep_job_validation(synthetic) -> None:
    s = Session(synthetic["ivf_flat"])
    with pytest.raises(ValueError, match="nprobe must be"):
        s.sweep_job(values=[NLIST + 1])
    with pytest.raises(ValueError, match=">= 1"):
        s.sweep_job(k=0)


def test_hnsw_layout_on_fresh_session_does_not_deadlock(synthetic) -> None:
    # hnsw_layout waits for the PCA job, whose thread needs other lazy values (source).
    # With one session-wide lock this deadlocked; run it in a thread with a timeout.
    s = Session(synthetic["hnsw_flat"])
    out = {}
    t = threading.Thread(target=lambda: out.setdefault("xy", s.hnsw_layout()), daemon=True)
    t.start()
    t.join(30)
    assert not t.is_alive(), "hnsw_layout deadlocked"
    assert out["xy"].shape == (N, 2)


def test_sweep_cache_reuses_ground_truth_and_bounds_entries(synthetic) -> None:
    session = Session(synthetic["ivf_flat"], vectors=synthetic["vectors"], disk_cache=False)
    first = session._sweep_data(10, 3, 0)
    assert session._sweep_data(10, 3, 0) is first
    for seed in range(1, 6):
        session._sweep_data(10, 3, seed)
    assert len(session._sweep_inputs) == 4
    assert (10, 3, 0) not in session._sweep_inputs
    again = session._sweep_data(10, 3, 0)
    np.testing.assert_array_equal(first[0].vectors, again[0].vectors)
    np.testing.assert_array_equal(first[1], again[1])


def test_sweep_cache_does_not_retain_oversized_arrays(synthetic, monkeypatch) -> None:
    monkeypatch.setattr("faissight.session._SWEEP_CACHE_BYTES", 1)
    session = Session(synthetic["ivf_flat"], vectors=synthetic["vectors"], disk_cache=False)
    queries, truth = session._sweep_data(10, 3, 0)
    assert len(queries) == 10
    assert truth.shape == (10, 3)
    assert not session._sweep_inputs


def test_sweep_keys_include_seed_and_repetitions(synthetic) -> None:
    session = Session(synthetic["ivf_flat"], vectors=synthetic["vectors"], disk_cache=False)
    a_id, a = session.sweep_job(values=[1], n_queries=10, repeats=1, seed=3)
    b_id, b = session.sweep_job(values=[1], n_queries=10, repeats=2, seed=3)
    c_id, c = session.sweep_job(values=[1], n_queries=10, repeats=1, seed=4)
    assert len({a_id, b_id, c_id}) == 3
    for job in (a, b, c):
        assert job.wait(10)
        assert job.is_done
    assert a.result.query_sha256 == b.result.query_sha256
    assert a.result.query_sha256 != c.result.query_sha256
    assert a.result.query_seed == 3
    assert b.result.repeats == 2


@pytest.mark.parametrize("bad", [np.empty((0, D)), np.full((1, D), np.nan)])
def test_invalid_queries_rejected_at_ingestion(synthetic, bad) -> None:
    with pytest.raises(InputError) as error:
        Session(synthetic["ivf_flat"], queries=bad)
    assert error.value.code == "QUERY_MISMATCH"
